from order_intake_ai.checks import line_issues, order_issues
from order_intake_ai.match import Candidate, Match
from order_intake_ai.synth import Customer


def cust(**kw):
    base = dict(id="C001", name="X", domain="x.fr", contact="A B", lang="fr", discount=0.1, own_codes=False,
                knows_skus=False, formats=["email"], po_format="PO{n}")
    return Customer(**{**base, **kw})


def auto(p):
    return Match("AUTO", "hybrid", p.sku, 0.9, 0.2, [Candidate(p.sku, p.name_en, 0.9)])


def codes(issues):
    return {i.code for i in issues}


def test_clean_line_has_no_issue(catalog):
    p = next(q for q in catalog if q.status == "active")
    c = cust()
    price = round(p.price * 0.9, 2)
    assert line_issues(auto(p), 2, price, c, {p.sku: p}, {"C001": {p.sku: [1, 2, 3]}}) == []


def test_price_and_quantity_flags(catalog):
    p = next(q for q in catalog if q.status == "active")
    by = {p.sku: p}
    c = cust()
    assert "price_mismatch" in codes(line_issues(auto(p), 2, p.price * 1.2, c, by, {}))
    assert "unusual_quantity" in codes(line_issues(auto(p), 100, None, c, by, {"C001": {p.sku: [5, 8, 10]}}))
    assert "missing_quantity" in codes(line_issues(auto(p), None, None, c, by, {}))


def test_discontinued_proposes_replacement(catalog):
    by = {q.sku: q for q in catalog}
    p = next(q for q in catalog if q.status == "discontinued")
    issues = line_issues(auto(p), 1, None, cust(), by, {})
    assert "discontinued" in codes(issues) and p.replaced_by in issues[0].message


def test_order_level_flags():
    c = cust()
    seen = {("C001", "PO-1")}
    assert "duplicate_po" in codes(order_issues(c, "a@x.fr", "PO-1", "2026-10-01", None, 2, seen))
    assert "unknown_customer" in codes(order_issues(None, "a@spam.com", "PO-2", None, None, 2, set()))
    assert "delivery_before_order" in codes(order_issues(c, "a@x.fr", "PO-3", "2026-10-05", "2026-10-01", 1, set()))
    assert "missing_po" in codes(order_issues(c, "a@x.fr", None, None, None, 1, set()))


def test_discontinued_flagged_on_review_line(catalog):
    by = {q.sku: q for q in catalog}
    p = next(q for q in catalog if q.status == "discontinued")
    m = Match("REVIEW", "hybrid", None, 0.8, 0.01, [Candidate(p.sku, p.name_en, 0.8)], "close alternatives")
    assert {"choose_product", "discontinued"} <= codes(line_issues(m, 1, None, cust(), by, {}))
