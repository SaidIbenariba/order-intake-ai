"""Synthetic product catalog of an office & janitorial supplies distributor (bilingual EN/FR).

Products come in families of near-identical siblings (A4 80g vs 90g, nitrile gloves M vs L, toner
26A vs 26X). That is what makes order matching hard in real life: the right product and the wrong
one share most of their words.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel

COLORS = {
    "blue": "bleu", "black": "noir", "red": "rouge", "green": "vert", "yellow": "jaune",
    "pink": "rose", "orange": "orange", "white": "blanc", "ivory": "ivoire", "grey": "gris",
    "purple": "violet", "transparent": "transparent", "cyan": "cyan", "magenta": "magenta",
}


class Product(BaseModel):
    sku: str
    mpn: str  # manufacturer part number
    name_en: str
    name_fr: str
    category: str
    brand: str
    attrs: dict[str, str]
    unit: str  # sales unit, e.g. "ream of 500 sheets"
    price: float  # list price, EUR, per sales unit
    status: str = "active"  # or "discontinued"
    replaced_by: str | None = None

    @property
    def search_text(self) -> str:
        return f"{self.name_en} | {self.name_fr}"


def _families(rng: random.Random):
    """Yield (category, brand, attrs, name_en, name_fr, unit, price, mpn)."""
    p = lambda lo, hi: round(rng.uniform(lo, hi), 2)
    code = lambda prefix: f"{prefix}{rng.randint(100000, 999999)}"

    # copy paper
    for brand, gsms in [("Navigator", [80, 90, 100]), ("Double A", [70, 80]), ("Clairefontaine", [80, 90, 100]),
                        ("Xerox Performer", [75, 80])]:
        for size in ["A4", "A3"]:
            for g in gsms:
                base = (4.2 if size == "A4" else 8.6) * (1 + (g - 80) / 100)
                yield ("paper", brand, {"size": size, "gsm": str(g), "color": "white"},
                       f"{brand} copy paper {size} {g}g white, ream of 500 sheets",
                       f"Ramette papier {brand} {size} {g}g blanc, 500 feuilles",
                       "ream of 500 sheets", p(base * 0.95, base * 1.25), code("PAP"))
    for c in ["yellow", "blue", "green", "pink", "ivory", "orange"]:
        yield ("paper", "Clairefontaine Trophee", {"size": "A4", "gsm": "80", "color": c},
               f"Clairefontaine Trophee coloured paper A4 80g {c}, ream of 500 sheets",
               f"Ramette papier couleur Clairefontaine Trophee A4 80g {COLORS[c]}, 500 feuilles",
               "ream of 500 sheets", p(7.5, 9.5), code("TRO"))

    # pens and markers
    for brand, model, kind_en, kind_fr, colors, pack in [
        ("Bic", "Cristal", "ballpoint pen medium", "stylo bille pointe moyenne", ["blue", "black", "red", "green"], 50),
        ("Bic", "Cristal Fine", "ballpoint pen fine", "stylo bille pointe fine", ["blue", "black", "red"], 50),
        ("Pilot", "G2", "gel pen 0.7", "stylo gel 0.7", ["blue", "black", "red", "green"], 12),
        ("Pilot", "BPS-GP", "ballpoint pen retractable", "stylo bille retractable", ["blue", "black", "red"], 12),
        ("Schneider", "Slider Memo XB", "ballpoint pen extra broad", "stylo bille extra large", ["blue", "black"], 10),
        ("Edding", "360", "whiteboard marker round tip", "marqueur tableau blanc pointe ogive", ["blue", "black", "red", "green"], 10),
        ("Sharpie", "Fine", "permanent marker fine", "marqueur permanent pointe fine", ["blue", "black", "red", "green"], 12),
        ("Stabilo", "Boss Original", "highlighter", "surligneur", ["yellow", "green", "pink", "orange", "blue"], 10),
    ]:
        for c in colors:
            yield ("writing", brand, {"model": model.lower(), "color": c},
                   f"{brand} {model} {kind_en} {c}, box of {pack}",
                   f"{kind_fr.capitalize()} {brand} {model} {COLORS[c]}, boite de {pack}",
                   f"box of {pack}", p(0.25 * pack, 0.6 * pack) if brand != "Bic" else p(9, 14), code(brand[:3].upper()))

    # lever arch files
    for brand in ["Esselte", "Leitz"]:
        for width in [50, 80]:
            for c in ["black", "blue", "red", "green", "yellow", "white"]:
                yield ("filing", brand, {"size": "A4", "width_mm": str(width), "color": c},
                       f"{brand} lever arch file A4 {width}mm {c}",
                       f"Classeur a levier {brand} A4 dos {width}mm {COLORS[c]}",
                       "each", p(2.4, 4.8), code(brand[:3].upper()))

    # toner cartridges (public manufacturer model codes)
    hp = [("26", "CF226", "black"), ("59", "CF259", "black"), ("85", "CE285", "black"), ("83", "CF283", "black"),
          ("78", "CE278", "black"), ("12", "Q2612", "black")]
    for model, mpn, c in hp:
        for v, mult in [("A", 1.0), ("X", 1.7)]:
            if model in ("12", "78", "85") and v == "X":
                continue
            yield ("toner", "HP", {"model": f"{model}{v}".lower(), "color": c},
                   f"HP {model}{v} original toner cartridge {c} ({mpn}{v})",
                   f"Toner HP {model}{v} original {COLORS[c]} ({mpn}{v})",
                   "each", p(70 * mult, 95 * mult), f"{mpn}{v}")
    for i, c in enumerate(["black", "cyan", "yellow", "magenta"]):
        for model, mpn_base in [("410A", "CF41"), ("207A", "W221")]:
            yield ("toner", "HP", {"model": model.lower(), "color": c},
                   f"HP {model} original toner cartridge {c} ({mpn_base}{i}A)",
                   f"Toner HP {model} original {COLORS[c]} ({mpn_base}{i}A)",
                   "each", p(85, 135), f"{mpn_base}{i}A")
    for model, c, lo in [("TN-2410", "black", 45), ("TN-2420", "black", 70), ("TN-243BK", "black", 50),
                         ("TN-243C", "cyan", 55), ("TN-243M", "magenta", 55), ("TN-243Y", "yellow", 55)]:
        yield ("toner", "Brother", {"model": model.lower(), "color": c},
               f"Brother {model} original toner cartridge {c}",
               f"Toner Brother {model} original {COLORS[c]}",
               "each", p(lo, lo * 1.3), model)
    for model, c in [("052", "black"), ("052H", "black"), ("054", "black"), ("054", "cyan"), ("054", "magenta"), ("054", "yellow")]:
        mpn = f"CRG-{model}{'' if c == 'black' else c[0].upper()}"
        yield ("toner", "Canon", {"model": model.lower(), "color": c},
               f"Canon {model} original toner cartridge {c} ({mpn})",
               f"Toner Canon {model} original {COLORS[c]} ({mpn})",
               "each", p(60, 120), mpn)

    # envelopes
    for fmt, dims, box in [("DL", "110x220mm", 500), ("C5", "162x229mm", 500), ("C4", "229x324mm", 250)]:
        for window in ["window", "no window"]:
            for closure_en, closure_fr in [("self-seal", "autocollante"), ("peel and seal", "bande protectrice")]:
                win_fr = "avec fenetre" if window == "window" else "sans fenetre"
                yield ("envelopes", "La Couronne", {"format": fmt.lower(), "window": "yes" if window == "window" else "no",
                                                    "closure": closure_en.split()[0]},
                       f"Envelopes {fmt} {dims} white {window} {closure_en}, box of {box}",
                       f"Enveloppes {fmt} {dims} blanches {win_fr} {closure_fr}, boite de {box}",
                       f"box of {box}", p(14, 38), code("LCO"))

    # staples, staplers, sticky notes, tape
    for size, box in [("24/6", 1000), ("24/6", 5000), ("26/6", 1000), ("26/6", 5000), ("23/8", 1000), ("23/10", 1000), ("23/13", 1000)]:
        yield ("stapling", "Rapid", {"staple": size, "pack": str(box)},
               f"Rapid staples {size} galvanised, box of {box}",
               f"Agrafes Rapid {size} galvanisees, boite de {box}",
               f"box of {box}", p(1.2, 2.5) * (box / 1000), code("RAP"))
    for model, sheets in [("F16", 16), ("S27", 30), ("HD110", 110)]:
        yield ("stapling", "Rapid", {"model": model.lower(), "sheets": str(sheets)},
               f"Rapid {model} stapler, {sheets} sheets capacity",
               f"Agrafeuse Rapid {model}, capacite {sheets} feuilles",
               "each", p(8, 60), code("RAP"))
    for size in ["76x76", "38x51", "76x127"]:
        for c in ["yellow", "pink", "blue", "green"]:
            if size != "76x76" and c in ("blue", "green"):
                continue
            yield ("notes", "Post-it", {"dims": size, "color": c},
                   f"Post-it notes {size}mm {c}, pack of 12 pads",
                   f"Notes Post-it {size}mm {COLORS[c]}, lot de 12 blocs",
                   "pack of 12", p(9, 18), code("PIT"))
    for name_en, name_fr, dims in [("Scotch Magic tape", "Ruban adhesif Scotch Magic invisible", "19mmx33m"),
                                   ("packaging tape brown", "ruban adhesif emballage marron", "48mmx66m"),
                                   ("packaging tape transparent", "ruban adhesif emballage transparent", "48mmx66m")]:
        yield ("tape", "Scotch", {"item": name_en.lower(), "dims": dims.lower()}, f"{name_en.capitalize()} {dims}, pack of 6",
               f"{name_fr.capitalize()} {dims}, lot de 6", "pack of 6", p(6, 16), code("SCO"))

    # notebooks
    for brand in ["Oxford", "Clairefontaine"]:
        for size in ["A4", "A5"]:
            for ruling_en, ruling_fr in [("lined", "ligne"), ("squared 5x5", "petits carreaux 5x5"), ("seyes", "grands carreaux seyes")]:
                for pages in [100, 180]:
                    yield ("notebooks", brand, {"size": size, "ruling": ruling_en.split()[0], "pages": str(pages)},
                           f"{brand} spiral notebook {size} {ruling_en} {pages} pages",
                           f"Cahier spirale {brand} {size} {ruling_fr} {pages} pages",
                           "each", p(1.8, 5.5), code(brand[:3].upper()))

    # labels
    for ref, per_sheet, dims in [("L7160", 21, "63.5x38.1mm"), ("L7163", 14, "99.1x38.1mm"), ("L7173", 10, "99.1x57mm"),
                                 ("L7165", 8, "99.1x67.7mm"), ("L7651", 65, "38.1x21.2mm"), ("L7167", 1, "199.6x289.1mm")]:
        yield ("labels", "Avery", {"model": ref.lower(), "per_sheet": str(per_sheet)},
               f"Avery {ref} laser labels {dims}, {per_sheet} per sheet, 100 sheets",
               f"Etiquettes laser Avery {ref} {dims}, {per_sheet} par feuille, 100 feuilles",
               "box of 100 sheets", p(18, 42), ref)

    # batteries
    for brand in ["Duracell", "Energizer"]:
        for size, packs in [("AA", [4, 12, 24]), ("AAA", [4, 12, 24]), ("9V", [1, 2]), ("C", [2]), ("D", [2])]:
            for pack in packs:
                yield ("batteries", brand, {"battery": size.lower(), "pack": str(pack)},
                       f"{brand} alkaline batteries {size}, pack of {pack}",
                       f"Piles alcalines {brand} {size}, lot de {pack}",
                       f"pack of {pack}", p(0.9 * pack, 1.4 * pack) + 2, code(brand[:3].upper()))

    # IT accessories
    for brand in ["SanDisk", "Kingston"]:
        for gb in [16, 32, 64, 128]:
            yield ("it", brand, {"capacity": f"{gb}gb"}, f"{brand} USB 3.0 flash drive {gb}GB",
                   f"Cle USB 3.0 {brand} {gb} Go", "each", p(5, 5 + gb * 0.12), code(brand[:3].upper()))
    for name_en, name_fr, layout in [("Logitech K120 keyboard USB", "Clavier Logitech K120 USB", "azerty"),
                                     ("Logitech K120 keyboard USB", "Clavier Logitech K120 USB", "qwerty")]:
        yield ("it", "Logitech", {"model": "k120", "layout": layout}, f"{name_en} {layout.upper()}",
               f"{name_fr} {layout.upper()}", "each", p(12, 18), code("LOG"))
    yield ("it", "Logitech", {"model": "m185"}, "Logitech M185 wireless mouse grey", "Souris sans fil Logitech M185 grise",
           "each", p(12, 17), code("LOG"))

    # janitorial: bin bags
    for brand in ["Raja", "Sacs Pro"]:
        for litres, colors in [(30, ["black", "white"]), (50, ["black", "blue"]), (100, ["black", "transparent"]),
                               (110, ["black"]), (130, ["black", "transparent"])]:
            for c in colors:
                roll = 20 if litres <= 50 else 10
                yield ("bags", brand, {"volume_l": str(litres), "color": c},
                       f"{brand} bin bags {litres}L {c}, roll of {roll}",
                       f"Sacs poubelle {brand} {litres}L {COLORS[c]}, rouleau de {roll}",
                       f"roll of {roll}", p(1.8, 4.5) * (litres / 50) + 1, code(brand[:3].upper()))

    # gloves
    for material_en, material_fr, colors in [("nitrile", "nitrile", ["blue", "black"]), ("latex", "latex", ["white"]),
                                             ("vinyl", "vinyle", ["transparent"])]:
        for c in colors:
            for s in ["S", "M", "L", "XL"]:
                yield ("gloves", "Mapa", {"material": material_en, "glove_size": s.lower(), "color": c},
                       f"Disposable {material_en} gloves powder-free {c} size {s}, box of 100",
                       f"Gants jetables {material_fr} non poudres {COLORS[c]} taille {s}, boite de 100",
                       "box of 100", p(5, 12), code("MAP"))

    # paper hygiene
    for item_en, item_fr, unit, lo, hi in [
        ("Toilet paper 2-ply, pack of 12 rolls", "Papier toilette 2 plis, lot de 12 rouleaux", "pack of 12", 4, 7),
        ("Toilet paper 2-ply, pack of 48 rolls", "Papier toilette 2 plis, lot de 48 rouleaux", "pack of 48", 15, 24),
        ("Jumbo toilet roll 2-ply 200m, pack of 6", "Papier toilette jumbo 2 plis 200m, lot de 6", "pack of 6", 14, 22),
        ("Kitchen towel rolls 2-ply, pack of 4", "Essuie-tout 2 plis, lot de 4 rouleaux", "pack of 4", 3, 6),
        ("Z-fold hand towels 2-ply, case of 3750", "Essuie-mains pliage Z 2 plis, carton de 3750", "case of 3750", 28, 42),
        ("Centrefeed wiping roll 2-ply 450 sheets, pack of 6", "Bobine devidage central 2 plis 450 formats, lot de 6", "pack of 6", 22, 34),
        ("Facial tissues box of 100", "Mouchoirs boite de 100", "each", 1, 2),
    ]:
        yield ("hygiene", "Tork" if "towel" in item_en.lower() or "wiping" in item_en.lower() else "Lotus",
               {"pack": unit.split()[-1]}, item_en, item_fr, unit, p(lo, hi), code("HYG"))

    # cleaning products
    for item_en, item_fr, sizes in [
        ("Liquid hand soap", "Savon liquide mains", ["500ml", "5l"]),
        ("Dishwashing liquid", "Liquide vaisselle", ["1l", "5l"]),
        ("Multi-surface cleaner", "Nettoyant multi-surfaces", ["1l", "5l"]),
        ("Bleach 2.6%", "Eau de javel 2,6%", ["1l", "5l"]),
        ("Glass cleaner spray", "Nettoyant vitres spray", ["750ml", "5l"]),
        ("Floor cleaner", "Nettoyant sols", ["1l", "5l"]),
        ("Hand sanitiser gel", "Gel hydroalcoolique", ["500ml", "1l", "5l"]),
    ]:
        for s in sizes:
            mult = {"500ml": 1, "750ml": 1.2, "1l": 1.5, "5l": 5}[s]
            yield ("cleaning", "Pro Clean", {"item": item_en.lower(), "volume": s}, f"{item_en} {s.upper() if s.endswith('l') and 'm' not in s else s}",
                   f"{item_fr} {s.upper() if s.endswith('l') and 'm' not in s else s}", "each", p(1.5 * mult, 3 * mult), code("PCL"))
    for c in ["blue", "red", "yellow", "green"]:
        yield ("cleaning", "Vileda", {"color": c}, f"Microfibre cloths 40x40cm {c}, pack of 10",
               f"Lavettes microfibre 40x40cm {COLORS[c]}, lot de 10", "pack of 10", p(8, 14), code("VIL"))

    # breakroom
    for item_en, item_fr, unit, lo, hi in [
        ("Coffee beans Lavazza Crema e Aroma 1kg", "Cafe en grains Lavazza Crema e Aroma 1kg", "each", 16, 22),
        ("Coffee beans Lavazza Qualita Oro 1kg", "Cafe en grains Lavazza Qualita Oro 1kg", "each", 17, 24),
        ("Ground coffee Carte Noire 250g", "Cafe moulu Carte Noire 250g", "each", 3.5, 5),
        ("Paper cups 20cl white, pack of 100", "Gobelets carton 20cl blancs, lot de 100", "pack of 100", 3, 6),
        ("Paper cups 20cl white, case of 1000", "Gobelets carton 20cl blancs, carton de 1000", "case of 1000", 25, 40),
        ("Plastic cups 20cl white, pack of 100", "Gobelets plastique 20cl blancs, lot de 100", "pack of 100", 2, 4),
        ("Sugar sticks 4g, box of 500", "Sucre buchettes 4g, boite de 500", "box of 500", 6, 10),
        ("Wooden stirrers 14cm, box of 1000", "Agitateurs bois 14cm, boite de 1000", "box of 1000", 3, 6),
        ("Tea bags Lipton Yellow Label, box of 100", "The Lipton Yellow Label, boite de 100 sachets", "box of 100", 5, 9),
        ("Mineral water 50cl, pack of 24", "Eau minerale 50cl, pack de 24", "pack of 24", 5, 9),
        ("Mineral water 1.5L, pack of 6", "Eau minerale 1,5L, pack de 6", "pack of 6", 2.5, 4.5),
    ]:
        yield ("breakroom", item_en.split()[2] if "Lavazza" in item_en else "Generic", {"item": item_en.lower()},
               item_en, item_fr, unit, p(lo, hi), code("BRK"))

    # desk accessories
    for item_en, item_fr in [
        ("Scissors 21cm stainless steel", "Ciseaux 21cm acier inoxydable"),
        ("Plastic ruler 30cm transparent", "Regle plastique 30cm transparente"),
        ("Hole punch 2 holes 25 sheets", "Perforatrice 2 trous 25 feuilles"),
        ("Desk calculator 12 digits Casio MS-20UC", "Calculatrice de bureau 12 chiffres Casio MS-20UC"),
        ("Correction tape Tipp-Ex Mini Pocket 5mmx6m", "Roller de correction Tipp-Ex Mini Pocket 5mmx6m"),
        ("Glue stick UHU 21g", "Baton de colle UHU 21g"),
        ("Paper clips 32mm, box of 100", "Trombones 32mm, boite de 100"),
        ("Rubber bands assorted 100g", "Elastiques assortis 100g"),
        ("Pencil HB Staedtler Noris, box of 12", "Crayon a papier HB Staedtler Noris, boite de 12"),
        ("Eraser Staedtler Mars plastic", "Gomme Staedtler Mars plastic"),
    ]:
        yield ("desk", "Generic", {"item": item_en.lower()}, item_en, item_fr, "each", p(1, 25), code("DSK"))


def generate_catalog(seed: int = 11, discontinued_share: float = 0.03) -> list[Product]:
    rng = random.Random(seed)
    products, used = [], set()
    for category, brand, attrs, en, fr, unit, price, mpn in _families(rng):
        sku = str(rng.randint(100000, 999999))
        while sku in used:
            sku = str(rng.randint(100000, 999999))
        used.add(sku)
        products.append(Product(sku=sku, mpn=mpn, name_en=en, name_fr=fr, category=category, brand=brand,
                                attrs=attrs, unit=unit, price=round(price, 2)))
    # discontinue a few products that have a true equivalent: same category and attributes, another brand.
    # That equivalent is the replacement a sales rep would propose.
    def equivalents(prod: Product) -> list[Product]:
        return [q for q in products if q.category == prod.category and q.attrs == prod.attrs
                and q.brand != prod.brand and q.status == "active"]

    pool = [q for q in products if equivalents(q)]
    for prod in rng.sample(pool, round(len(products) * discontinued_share)):
        if equivalents(prod):
            prod.status, prod.replaced_by = "discontinued", rng.choice(equivalents(prod)).sku
    return products


def save_catalog(products: list[Product], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([p.model_dump() for p in products], indent=1, ensure_ascii=False))


def load_catalog(path: Path) -> list[Product]:
    return [Product(**d) for d in json.loads(path.read_text())]
