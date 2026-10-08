"""Inbox folder of .eml orders -> extracted, matched, checked orders -> ERP-ready export + review queue."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .catalog import Product, load_catalog
from .checks import line_issues, order_issues
from .evaluate import evaluate_orders
from .export import write_outputs
from .extract import DEFAULT_MODEL, ExtractedOrder, extract, read_eml, source_text, to_order
from .match import Matcher, Memory
from .synth import Customer


def load_customers(path: Path) -> list[Customer]:
    return [Customer(**d) for d in json.loads(path.read_text())]


def load_cached(path: Path, eml: Path) -> ExtractedOrder:
    """Cached model output, re-normalized with the current code (so normalization fixes apply without a re-run)."""
    d = json.loads(path.read_text())
    mail = read_eml(eml)
    route, text = source_text(mail)
    return to_order(d["raw"], mail, route, d["seconds"], text)


def process_order(e: ExtractedOrder, file: str, matcher: Matcher, memory: Memory | None, customers: dict[str, Customer],
                  history: dict, seen_pos: set[tuple[str, str]]) -> dict:
    domain = e.sender.rsplit("@", 1)[-1]
    customer = next((c for c in customers.values() if c.domain == domain), None)
    by_sku: dict[str, Product] = matcher.by_sku
    lines = []
    for i, ln in enumerate(e.lines, 1):
        m = matcher.match(ln.description, customer.id if customer else None, ln.customer_ref, memory)
        issues = line_issues(m, ln.quantity, ln.unit_price, customer, by_sku, history, ln.qty_note)
        p = by_sku.get(m.sku) if m.sku else None
        lines.append({
            "line_no": i, "customer_ref": ln.customer_ref, "description": ln.description,
            "quantity": ln.quantity, "unit_price_written": ln.unit_price,
            "match": {"status": m.status, "method": m.method, "sku": m.sku, "name": p.name_en if p else None,
                      "unit": p.unit if p else None, "score": round(m.score, 3), "margin": round(m.margin, 3),
                      "reason": m.reason,
                      "candidates": [{"sku": c.sku, "name": c.name, "score": c.score} for c in m.candidates[:3]]},
            "price": round(p.price * (1 - customer.discount), 2) if p and customer else None,
            "issues": [asdict(x) for x in issues],
        })
    o_issues = order_issues(customer, e.sender, e.po_number, e.order_date, e.delivery_date, len(e.lines), seen_pos)
    if customer and e.po_number:
        seen_pos.add((customer.id, e.po_number))
    flagged = o_issues or any(ln["issues"] for ln in lines)
    return {
        "file": file, "sender": e.sender, "customer_id": customer.id if customer else None,
        "customer_name": customer.name if customer else None, "po_number": e.po_number,
        "order_date": e.order_date, "delivery_date": e.delivery_date, "route": e.route,
        "seconds": round(e.seconds, 1), "status": "REVIEW" if flagged else "APPROVED",
        "issues": [asdict(x) for x in o_issues], "lines": lines,
    }


def run(in_dir: Path, out_dir: Path, data_dir: Path, truth_dir: Path | None = None, model: str = DEFAULT_MODEL,
        use_memory: bool = True, log: Callable[[str], None] = print, matcher: Matcher | None = None) -> dict:
    catalog = load_catalog(data_dir / "catalog.json")
    customers = {c.id: c for c in load_customers(data_dir / "customers.json")}
    history = json.loads((data_dir / "history.json").read_text())
    cfg = json.loads((data_dir / "matcher.json").read_text())
    if matcher is None:
        log("Loading catalog and embedding model...")
        matcher = Matcher(catalog, tau=cfg["tau"], margin=cfg["margin"])
    memory = Memory(data_dir / "memory.json") if use_memory else None

    cache = out_dir / "extractions"
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.eml"))
    orders, seen_pos = [], set()
    for k, f in enumerate(files, 1):
        cached = cache / f"{f.stem}.{model.replace(':', '_')}.json"
        if cached.exists():
            e = load_cached(cached, f)
        else:
            e = extract(f, model=model)
            cached.write_text(json.dumps(asdict(e), indent=1, ensure_ascii=False))
        o = process_order(e, f.name, matcher, memory, customers, history, seen_pos)
        orders.append(o)
        log(f"[{k}/{len(files)}] {f.name}: {o['status']} ({e.route}, {e.seconds:.0f}s, {len(o['lines'])} lines)")

    report = {"model": model, "memory": use_memory, "tau": matcher.tau, "margin": matcher.margin, "orders": orders}
    if truth_dir and truth_dir.exists():
        report["summary"] = evaluate_orders(orders, truth_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    write_outputs(report, out_dir)
    return report
