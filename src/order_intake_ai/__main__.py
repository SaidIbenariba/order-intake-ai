"""CLI.

uv run python -m order_intake_ai setup           # catalog, customers, past orders, tuned thresholds, memory, sample inbox
uv run python -m order_intake_ai eval-matching   # line-level matching test on 150 held-out orders (no LLM, ~1 min)
uv run python -m order_intake_ai inbox --seed N  # write a new synthetic inbox (default: data/samples)
uv run python -m order_intake_ai run             # process the sample inbox end to end with the local model
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .catalog import generate_catalog, load_catalog, save_catalog
from .evaluate import run_lines, summarize_lines, tune
from .match import Matcher, Memory
from .render import write_batch
from .synth import generate_orders, make_customers, make_history

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT = ROOT / "data", ROOT / "out"
DEV_SEED, HELDOUT_SEED, INBOX_SEED = 1, 2027, 2029  # 2028 = end-to-end dev inbox (data/dev_inbox)
PRIVATE = {"habitual", "own_code_map", "usual_qty"}  # generator state, not customer master data


def setup(n_inbox: int) -> None:
    catalog = generate_catalog()
    customers = make_customers(catalog)
    save_catalog(catalog, DATA / "catalog.json")
    (DATA / "customers.json").write_text(json.dumps([c.model_dump(exclude=PRIVATE) for c in customers], indent=1, ensure_ascii=False))
    (DATA / "history.json").write_text(json.dumps(make_history(customers), indent=1))
    print(f"catalog: {len(catalog)} products, {len(customers)} customers")

    past = generate_orders(catalog, customers, 150, "dev", DEV_SEED)
    matcher = Matcher(catalog)
    tau, margin, info = tune(run_lines(matcher, past))
    (DATA / "matcher.json").write_text(json.dumps({"tau": tau, "margin": margin, "tuned_on": f"dev seed {DEV_SEED}", **info}, indent=1))
    print(f"thresholds tuned on {sum(len(o.lines) for o in past)} dev lines: tau={tau} margin={margin} {info}")

    if (DATA / "memory.json").exists():
        (DATA / "memory.json").unlink()
    memory = Memory(DATA / "memory.json")
    for o in past:  # staff confirmed these past orders
        for ln in o.lines:
            if ln.sku:
                memory.learn(o.customer_id, ln.text, ln.customer_ref, ln.sku)
    memory.save()
    print(f"memory: {len(memory.pairs)} confirmed customer wordings from past orders")

    make_inbox(DATA / "samples", n_inbox, INBOX_SEED)


def make_inbox(inbox: Path, n: int, seed: int) -> None:
    catalog = load_catalog(DATA / "catalog.json")
    customers = make_customers(catalog)
    if inbox.exists():
        shutil.rmtree(inbox)
    write_batch(generate_orders(catalog, customers, n, "heldout", seed), customers, inbox, seed=seed)
    print(f"inbox: {n} order emails in {inbox} (seed {seed})")


def eval_matching() -> None:
    from .synth import make_customers as mk

    catalog = load_catalog(DATA / "catalog.json")
    customers = mk(catalog)
    cfg = json.loads((DATA / "matcher.json").read_text())
    matcher = Matcher(catalog, tau=cfg["tau"], margin=cfg["margin"])
    memory = Memory(DATA / "memory.json")
    report = {"thresholds": cfg}
    for name, pool, seed in [("dev", "dev", DEV_SEED), ("heldout", "heldout", HELDOUT_SEED)]:
        orders = generate_orders(catalog, customers, 150, pool, seed)
        for mode, mem in [("no_memory", None), ("with_memory", memory)]:
            if name == "dev" and mode == "with_memory":
                continue  # the memory was built from these orders
            s = summarize_lines(run_lines(matcher, orders, mem))
            report[f"{name}_{mode}"] = s
            print(f"{name:8} {mode:12} lines={s['lines']} auto={s['auto_rate']:.1%} wrong={s['wrong_auto_matches']} "
                  f"top1(clear)={s['top1_clear']:.1%} top3(clear)={s['top3_clear']:.1%} "
                  f"not-in-catalog auto={s['not_in_catalog_auto_matched']}/{s['not_in_catalog_lines']}")
    OUT.mkdir(exist_ok=True)
    (OUT / "matching_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(prog="order_intake_ai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--inbox", type=int, default=30)
    sub.add_parser("eval-matching")
    i = sub.add_parser("inbox")
    i.add_argument("--seed", type=int, default=INBOX_SEED)
    i.add_argument("--n", type=int, default=30)
    i.add_argument("--out", default=str(DATA / "samples"))
    r = sub.add_parser("run")
    r.add_argument("--in", dest="in_dir", default=str(DATA / "samples"))
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--model", default="qwen2.5vl")
    r.add_argument("--no-memory", action="store_true")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup(a.inbox)
    elif a.cmd == "eval-matching":
        eval_matching()
    elif a.cmd == "inbox":
        make_inbox(Path(a.out), a.n, a.seed)
    else:
        from .pipeline import run

        in_dir = Path(a.in_dir)
        rep = run(in_dir, Path(a.out), DATA, in_dir / "truth", model=a.model, use_memory=not a.no_memory)
        if "summary" in rep:
            s = {k: v for k, v in rep["summary"].items() if k != "errors"}
            print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
