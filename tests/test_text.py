from order_intake_ai.text import compact_code, compare_attrs, normalize, parse_attrs


def test_same_attributes_in_english_and_french():
    en = parse_attrs("Copy paper A4 80gsm white, ream of 500 sheets")
    fr = parse_attrs("Ramette papier A4 80 g/m2 blanche, 500 feuilles")
    for key in ("size", "grams", "color", "pack"):
        assert en[key] == fr[key]


def test_units_are_converted():
    assert parse_attrs("classeur levier dos 8 cm")["mm"] == {"80"}
    assert parse_attrs("hand soap 500ml")["litres"] == {"0.5"}
    assert parse_attrs("sacs 100-litre")["litres"] == {"100"}
    assert parse_attrs("cafe 1kg")["grams"] == {"1000"}


def test_codes_and_models():
    assert parse_attrs("HP 26A noir")["model"] == {"26a"}
    assert parse_attrs("hp26x")["model"] == {"26x"}
    assert parse_attrs("Brother TN 2420")["model"] == {"tn-2420"}
    assert compact_code("tn-2420") == "TN2420"


def test_whiteboard_is_not_a_colour():
    assert parse_attrs("marqueur tableau blanc noir")["color"] == {"black"}


def test_glove_sizes_and_batteries():
    assert parse_attrs("gants nitrile taille M")["glove_size"] == {"m"}
    assert parse_attrs("nitrile gloves blue m, box of 100")["glove_size"] == {"m"}
    assert parse_attrs("piles LR6 x24")["battery"] == {"aa"}
    assert parse_attrs("duracell batteries d, pack of 2")["battery"] == {"d"}


def test_envelopes():
    a = parse_attrs("env DL plain peel & seal")
    assert a["window"] == {"no"} and a["closure"] == {"peel"}
    assert parse_attrs("enveloppes C5 avec fenetre autocollante")["window"] == {"yes"}


def test_compare_attrs_counts_conflicts():
    line = parse_attrs("A4 160g paper")
    product = parse_attrs("Navigator copy paper A4 100g white, ream of 500 sheets")
    agree, conflict, missing = compare_attrs(line, product)
    assert agree == 1 and conflict == 1 and missing == 0


def test_normalize_expands_abbreviations():
    assert normalize("Bic pens BLK, bx 50") == "bic pens black box 50"
