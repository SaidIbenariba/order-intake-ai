"""Business checks on a matched order. Any issue sends the order to review instead of the ERP."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Product
from .match import Match
from .synth import Customer

PRICE_TOL = 0.02  # 2 % difference between the customer's price and our price list
QTY_FACTOR = 3  # more than 3x the largest quantity this customer ever ordered for this product


@dataclass
class Issue:
    code: str
    message: str


def line_issues(m: Match, quantity: float | None, unit_price: float | None, customer: Customer | None,
                by_sku: dict[str, Product], history: dict[str, dict[str, list[float]]],
                qty_note: str | None = None) -> list[Issue]:
    out = []
    if qty_note:
        out.append(Issue("quantity_unclear", f"{qty_note}. Check the quantity."))
    if m.status == "NO_MATCH":
        out.append(Issue("not_in_catalog", "No product close enough in the catalog. New item, or a product we don't sell?"))
    elif m.status == "REVIEW":
        out.append(Issue("choose_product", f"Pick the product: {m.reason}"))
    p = by_sku.get(m.sku) if m.sku else None
    if m.status == "REVIEW":  # flag a discontinued close candidate before a human picks the product
        close = [by_sku.get(c.sku) for c in m.candidates if c.score >= m.candidates[0].score - 0.1]
        for q in close:
            if q and q.status == "discontinued":
                repl = by_sku.get(q.replaced_by)
                out.append(Issue("discontinued", f"Candidate {q.sku} is discontinued. Replacement: {repl.sku} {repl.name_en}"
                                 if repl else f"Candidate {q.sku} is discontinued."))
                break
    if p and p.status == "discontinued":
        repl = by_sku.get(p.replaced_by)
        out.append(Issue("discontinued", f"Discontinued. Proposed replacement: {repl.sku} {repl.name_en}" if repl else "Discontinued."))
    if quantity is None or quantity <= 0:
        out.append(Issue("missing_quantity", "Quantity missing or not a number."))
    if p and customer and unit_price is not None:
        ours = round(p.price * (1 - customer.discount), 2)
        if abs(unit_price - ours) > PRICE_TOL * ours:
            out.append(Issue("price_mismatch", f"Customer wrote {unit_price:.2f}, their price is {ours:.2f}."))
    if p and customer and quantity:
        past = history.get(customer.id, {}).get(p.sku)
        if past and quantity > QTY_FACTOR * max(past):
            out.append(Issue("unusual_quantity", f"Ordered {quantity:g}; usually {min(past):g} to {max(past):g}. Typo?"))
    return out


def order_issues(customer: Customer | None, sender: str, po: str | None, order_date: str | None,
                 delivery_date: str | None, n_lines: int, seen_pos: set[tuple[str, str]]) -> list[Issue]:
    out = []
    if customer is None:
        out.append(Issue("unknown_customer", f"Sender {sender} is not a known customer."))
    if not po:
        out.append(Issue("missing_po", "No PO number found."))
    elif customer and (customer.id, po) in seen_pos:
        out.append(Issue("duplicate_po", f"PO {po} was already received from this customer. Duplicate order?"))
    if order_date and delivery_date and delivery_date < order_date:
        out.append(Issue("delivery_before_order", "Requested delivery date is before the order date."))
    if n_lines == 0:
        out.append(Issue("no_lines", "No order lines found."))
    return out
