from order_intake_ai.match import Memory


def by_name(catalog, name):
    return next(p for p in catalog if p.name_en == name)


def test_our_sku_in_line_is_an_exact_match(matcher, catalog):
    p = catalog[10]
    m = matcher.match(f"Ref {p.sku} something")
    assert m.status == "AUTO" and m.method == "code" and m.sku == p.sku


def test_manufacturer_code_with_space(matcher, catalog):
    p = by_name(catalog, "Brother TN-2420 original toner cartridge black")
    m = matcher.match("toner TN 2420")
    assert m.method == "code" and m.sku == p.sku


def test_toner_a_and_x_are_not_confused(matcher, catalog):
    a = by_name(catalog, "HP 26A original toner cartridge black (CF226A)")
    m = matcher.match("HP 26A toner black")
    assert m.candidates[0].sku == a.sku


def test_near_miss_product_is_not_auto_matched(matcher):
    for text in ["Copy paper A4 160g white ream", "Nitrile gloves blue size XXL box 100", "Bin bags 240L black"]:
        assert matcher.match(text).status != "AUTO", text


def test_contradicted_attribute_is_reported(matcher):
    cands = matcher.score_all("Navigator copy paper A4 160g white")
    navigator_100 = next(c for c in cands if c.name.startswith("Navigator copy paper A4 100g"))
    assert "grams" in navigator_100.conflicts


def test_memory_resolves_customer_wording(matcher, catalog, tmp_path):
    p = by_name(catalog, "Raja bin bags 50L black, roll of 20")
    mem = Memory(tmp_path / "m.json")
    text = "sacs noirs grand modele"
    assert matcher.match(text, "C001", None, mem).status != "AUTO" or matcher.match(text, "C001", None, mem).sku == p.sku
    mem.learn("C001", text, None, p.sku)
    mem.save()
    m = matcher.match(text, "C001", None, Memory(tmp_path / "m.json"))
    assert m.status == "AUTO" and m.method == "memory" and m.sku == p.sku
    assert matcher.match(text, "C002", None, mem).method != "memory"  # memory is per customer


def test_memory_by_customer_article_code(matcher, catalog):
    p = by_name(catalog, "Raja bin bags 50L black, roll of 20")
    mem = Memory()
    mem.learn("C002", "whatever", "ART-1234", p.sku)
    assert matcher.match("totally different wording", "C002", "art 1234", mem).sku == p.sku


def test_code_in_customer_ref_field(matcher, catalog):
    p = next(q for q in catalog if q.mpn == "L7160")
    assert matcher.match("sticky labels 63.5x38.1mm", customer_ref="AVERY L7160").sku == p.sku
