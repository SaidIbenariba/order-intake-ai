"""Line-level matching metrics against ground truth, threshold tuning, and end-to-end order metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .match import Match, Matcher, Memory
from .synth import TruthOrder


@dataclass
class LineOutcome:
    order_id: str
    text: str
    kind: str
    ambiguous: bool
    truth: str | None
    status: str
    method: str
    pred: str | None
    top3: list[str]
    score: float
    margin: float
    second: float

    @property
    def wrong_auto(self) -> bool:
        return self.status == "AUTO" and self.pred != self.truth


def outcome(order_id: str, text: str, kind: str, ambiguous: bool, truth: str | None, m: Match) -> LineOutcome:
    top3 = [c.sku for c in m.candidates[:3]]
    second = m.candidates[1].score if len(m.candidates) > 1 else 0.0
    return LineOutcome(order_id, text, kind, ambiguous, truth, m.status, m.method, m.sku, top3,
                       m.score, m.margin, second)


def run_lines(matcher: Matcher, orders: list[TruthOrder], memory: Memory | None = None) -> list[LineOutcome]:
    """Match every truth line (the matcher sees the text exactly as the customer wrote it)."""
    res = []
    for o in orders:
        for ln in o.lines:
            m = matcher.match(ln.text, o.customer_id, ln.customer_ref, memory)
            res.append(outcome(o.order_id, ln.text, ln.kind, ln.ambiguous, ln.sku, m))
    return res


def summarize_lines(rows: list[LineOutcome]) -> dict:
    n = len(rows)
    in_cat = [r for r in rows if r.truth is not None]
    clear = [r for r in in_cat if not r.ambiguous]
    auto = [r for r in rows if r.status == "AUTO"]
    wrong = [r for r in auto if r.wrong_auto]

    def top(r: LineOutcome, k: int) -> bool:
        return r.truth == r.pred if r.method in ("code", "memory") else r.truth in r.top3[:k]

    return {
        "lines": n,
        "auto_matched": len(auto),
        "auto_rate": len(auto) / n if n else 0,
        "wrong_auto_matches": len(wrong),
        "auto_precision": (len(auto) - len(wrong)) / len(auto) if auto else 1.0,
        "top1_clear": sum(top(r, 1) for r in clear) / len(clear) if clear else 0,
        "top3_clear": sum(top(r, 3) for r in clear) / len(clear) if clear else 0,
        "top3_all_in_catalog": sum(top(r, 3) for r in in_cat) / len(in_cat) if in_cat else 0,
        "ambiguous_lines": sum(r.ambiguous for r in rows),
        "ambiguous_auto_matched": sum(r.ambiguous and r.status == "AUTO" for r in rows),
        "not_in_catalog_lines": sum(r.truth is None for r in rows),
        "not_in_catalog_auto_matched": sum(r.truth is None and r.status == "AUTO" for r in rows),
        "by_method": {m: sum(r.method == m and r.status == "AUTO" for r in rows) for m in ("code", "memory", "hybrid")},
        "wrong_examples": [asdict(r) for r in wrong[:10]],
    }


def tune(rows: list[LineOutcome], max_wrong: int = 0) -> tuple[float, float, dict]:
    """Pick (tau, margin) with the most hybrid auto-matches and at most `max_wrong` wrong ones.

    Only hybrid lines are affected; code and memory matches don't use the thresholds.
    """
    hyb = [r for r in rows if r.method in ("hybrid", "none")]
    best = (0.99, 0.5, -1, 0)
    for t100 in range(40, 96):
        tau = t100 / 100
        for m100 in range(0, 31):
            margin = m100 / 100
            auto = [r for r in hyb if r.score >= tau and r.margin >= margin]
            wrong = sum(r.top3[0] != r.truth for r in auto)
            if wrong <= max_wrong and len(auto) > best[2]:
                best = (tau, margin, len(auto), wrong)
    tau, margin, n_auto, n_wrong = best
    return tau, margin, {"hybrid_lines": len(hyb), "hybrid_auto": n_auto, "hybrid_wrong": n_wrong}


# ---------- end to end: extracted + matched orders vs truth ----------

ANOMALY_CODES = {
    "qty_outlier": {"unusual_quantity"},
    "price_mismatch": {"price_mismatch"},
    "discontinued": {"discontinued"},
    "not_in_catalog": {"not_in_catalog", "choose_product"},
    "duplicate_po": {"duplicate_po"},
}


def _align(truth_texts: list[str], got_texts: list[str]) -> dict[int, int]:
    """truth line index -> extracted line index."""
    from rapidfuzz import fuzz

    from .text import normalize

    if len(truth_texts) == len(got_texts):
        return {i: i for i in range(len(truth_texts))}
    pairs = sorted(((fuzz.ratio(normalize(t), normalize(g)), i, j) for i, t in enumerate(truth_texts)
                    for j, g in enumerate(got_texts)), reverse=True)
    out, used = {}, set()
    for s, i, j in pairs:
        if s >= 60 and i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def evaluate_orders(orders: list[dict], truth_dir) -> dict:
    import json
    from pathlib import Path

    from .text import compact_code, normalize

    n_lines = found = desc_exact = qty_ok = price_ok = auto = wrong_auto = 0
    head = {"po_number": 0, "order_date": 0, "delivery_date": 0}
    approved = approved_correct = 0
    anomalies = {k: [0, 0] for k in ANOMALY_CODES}
    clean_total = clean_approved = 0
    per_route: dict[str, list[float]] = {}
    errors = []
    for o in orders:
        tp = Path(truth_dir) / f"{Path(o['file']).stem}.json"
        if not tp.exists():
            continue
        t = TruthOrder(**json.loads(tp.read_text()))
        per_route.setdefault(o["route"], []).append(o["seconds"])
        po_ok = compact_code(o["po_number"] or "") == compact_code(t.po_number)
        head["po_number"] += po_ok
        head["order_date"] += o["order_date"] == t.order_date
        dd_ok = o["delivery_date"] == t.delivery_date
        head["delivery_date"] += dd_ok
        align = _align([ln.text for ln in t.lines], [ln["description"] for ln in o["lines"]])
        order_ok = po_ok and dd_ok and len(o["lines"]) == len(t.lines)
        for i, tl in enumerate(t.lines):
            n_lines += 1
            j = align.get(i)
            if j is None:
                order_ok = False
                continue
            gl = o["lines"][j]
            found += 1
            with_ref = f"{gl['customer_ref'] or ''} {gl['description']}"  # "Ref 537416 ..." split into ref + description
            desc_exact += normalize(tl.text) in (normalize(gl["description"]), normalize(with_ref))
            q = gl["quantity"] is not None and abs(gl["quantity"] - tl.quantity) < 1e-6
            pr = (tl.unit_price is None and gl["unit_price_written"] is None) or (
                tl.unit_price is not None and gl["unit_price_written"] is not None and abs(gl["unit_price_written"] - tl.unit_price) < 0.005)
            qty_ok += q
            price_ok += pr
            m = gl["match"]
            if m["status"] == "AUTO":
                auto += 1
                if m["sku"] != tl.sku:
                    wrong_auto += 1
                    errors.append({"file": o["file"], "text": tl.text, "extracted": gl["description"], "truth": tl.sku, "got": m["sku"]})
            order_ok = order_ok and q and pr and m["sku"] == tl.sku
        if o["status"] == "APPROVED":
            approved += 1
            approved_correct += order_ok
            if not order_ok:
                errors.append({"file": o["file"], "approved_with_error": True})
        if t.anomaly:
            codes = {x["code"] for x in o["issues"]} | {x["code"] for ln in o["lines"] for x in ln["issues"]}
            anomalies[t.anomaly][0] += 1
            anomalies[t.anomaly][1] += o["status"] == "REVIEW" and bool(codes & ANOMALY_CODES[t.anomaly])
        else:
            clean_total += 1
            clean_approved += o["status"] == "APPROVED"
    n = sum(len(v) for v in per_route.values())
    return {
        "orders": n,
        "approved": approved,
        "approved_fully_correct": approved_correct,
        "errors_reaching_erp": approved - approved_correct,
        "clean_orders": clean_total,
        "clean_orders_approved": clean_approved,
        "anomalies_total": sum(v[0] for v in anomalies.values()),
        "anomalies_caught": sum(v[1] for v in anomalies.values()),
        "anomalies_by_type": {k: f"{v[1]}/{v[0]}" for k, v in anomalies.items() if v[0]},
        "header_accuracy": {k: v / n for k, v in head.items()} if n else {},
        "lines": n_lines,
        "lines_extracted": found,
        "description_exact": desc_exact / n_lines if n_lines else 0,
        "quantity_accuracy": qty_ok / n_lines if n_lines else 0,
        "price_accuracy": price_ok / n_lines if n_lines else 0,
        "lines_auto_matched": auto,
        "line_auto_rate": auto / n_lines if n_lines else 0,
        "wrong_auto_matches": wrong_auto,
        "avg_seconds_by_route": {k: round(sum(v) / len(v), 1) for k, v in per_route.items()},
        "errors": errors[:20],
    }
