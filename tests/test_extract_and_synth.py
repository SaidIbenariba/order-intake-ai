import random

from order_intake_ai.extract import EmailOrder, _grounded, clean_line, clean_po, parse_date, parse_number, read_eml, to_order
from order_intake_ai.render import write_batch
from order_intake_ai.synth import generate_orders, is_ambiguous, make_customers


def test_numbers_and_dates():
    assert parse_number("1 234,50") == 1234.5
    assert parse_number("12.50") == 12.5
    assert parse_number("x3") == 3
    assert parse_date("08/10/2026") == "2026-10-08"
    assert parse_date("8 Oct 2026") == "2026-10-08"
    assert parse_date("soon") is None


def test_po_label_is_removed():
    assert clean_po("PO: LA-9756") == "LA-9756"
    assert clean_po("Bon de commande n° BC-2026-0012") == "BC-2026-0012"
    assert clean_po("HAM/2026/184") == "HAM/2026/184"
    assert clean_po("PO-2026-03080") == "PO-2026-03080"
    assert clean_po("PO 4500008100") == "4500008100"
    assert clean_po("PO#4471") == "4471"


def test_to_order_falls_back_to_email_date():
    mail = EmailOrder(sender="a@x.fr", subject="", sent="2026-10-01", body="")
    raw = {"po_number": "PO 12", "order_date": "", "delivery_date": "", "lines": [
        {"customer_ref": "", "description": "papier A4", "quantity": "10", "unit_price": ""},
        {"customer_ref": "", "description": "", "quantity": "", "unit_price": ""}]}
    o = to_order(raw, mail, "email-body", 1.0)
    assert o.order_date == "2026-10-01" and len(o.lines) == 1 and o.lines[0].quantity == 10


def test_generator_is_deterministic(catalog):
    cs = make_customers(catalog)
    a = generate_orders(catalog, cs, 10, "dev", 1)
    b = generate_orders(catalog, cs, 10, "dev", 1)
    assert [o.model_dump() for o in a] == [o.model_dump() for o in b]


def test_ambiguity_label(catalog):
    p = next(q for q in catalog if q.name_en.startswith("Navigator copy paper A4 80g"))
    assert is_ambiguous("copy paper A4 80g white", p, catalog)
    assert not is_ambiguous("Navigator copy paper A4 80g", p, catalog)


def test_rendered_email_roundtrip(catalog, tmp_path):
    cs = make_customers(catalog)
    orders = generate_orders(catalog, cs, 6, "heldout", 5)
    write_batch(orders, cs, tmp_path, seed=5)
    for o in orders:
        mail = read_eml(tmp_path / f"{o.order_id}.eml")
        assert mail.sender == o.sender
        assert (mail.attachment is not None) == (o.format in ("pdf", "photo"))


def test_quantity_copied_into_description_is_removed():
    assert clean_line("10 Pencil HB Staedtler Noris, x12", 10) == ("Pencil HB Staedtler Noris, x12", None)
    assert clean_line("cahier A4 180 pages 54", 54) == ("cahier A4 180 pages", None)
    assert clean_line("12a toner cartridge black", 10) == ("12a toner cartridge black", None)


def test_leading_number_that_disagrees_is_flagged():
    desc, note = clean_line("1 feutre permanent fine bleus, 12 par boite", 12)
    assert desc.startswith("1 feutre") and "starts with 1" in note


def test_value_not_in_source_is_dropped():
    src = "PO: LA-1\nDate: 20/09/2026"
    assert _grounded("20/09/2026", src) == "20/09/2026"
    assert _grounded("08/09/2026", src) is None
    assert _grounded("08/09/2026", None) == "08/09/2026"  # vision: can't check


def test_to_order_drops_invented_delivery_date():
    mail = EmailOrder(sender="a@x.fr", subject="", sent="2026-09-01", body="")
    raw = {"po_number": "PO-7", "order_date": "01/09/2026", "delivery_date": "05/09/2026", "lines": []}
    o = to_order(raw, mail, "pdf-text", 1.0, "PO-7 Date: 01/09/2026")
    assert o.order_date == "2026-09-01" and o.delivery_date is None and o.po_number == "PO-7"


def test_only_date_without_delivery_label_is_the_order_date():
    mail = EmailOrder(sender="a@x.fr", subject="", sent="2026-10-01", body="")
    raw = {"po_number": "NGA1", "order_date": "", "delivery_date": "06/10/2026", "lines": []}
    o = to_order(raw, mail, "pdf-text", 1.0, "PO Number: NGA1  Date: 06/10/2026")
    assert o.order_date == "2026-10-06" and o.delivery_date is None
    o = to_order(raw, mail, "pdf-text", 1.0, "PO Number: NGA1  Delivery date: 06/10/2026")
    assert o.delivery_date == "2026-10-06"
