"""Synthetic customers, order history and purchase orders, with ground truth for every line.

How customers write orders is the hard part, so the generator models it:
- each customer has habitual products written the same way every time (repeat orders)
- new products are written freshly: synonyms, French or English, abbreviations, dropped brand or
  pack size, typos, attribute formats like "80gsm" / "80 g/m2" / "5 cm"
- the held-out batch draws from wording pools never used in the dev batch ("bin liners", "LR6",
  "medium"), so the measured accuracy is not just the matcher memorising the generator
- planted problems: quantity typos, outdated prices, discontinued products, near-miss products that
  are not in the catalog (A4 160g, gloves XXL, HP 30A), duplicate PO numbers
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta

from pydantic import BaseModel

from .catalog import Product

ANOMALIES = ["qty_outlier", "price_mismatch", "discontinued", "not_in_catalog", "duplicate_po"]


class Customer(BaseModel):
    id: str
    name: str
    domain: str
    contact: str
    lang: str  # "en" or "fr"
    discount: float
    own_codes: bool  # sends its own article codes (buyer has an ERP)
    knows_skus: bool  # sometimes quotes our product codes
    formats: list[str]  # email | pdf | photo
    po_format: str
    habitual: dict[str, str] = {}  # sku -> fixed wording
    own_code_map: dict[str, str] = {}  # sku -> customer's article code
    usual_qty: dict[str, int] = {}


class TruthLine(BaseModel):
    text: str
    customer_ref: str | None = None
    quantity: float
    unit_price: float | None = None
    sku: str | None  # None = not in our catalog
    kind: str  # repeat | new | sku_code | mpn_code | not_in_catalog
    ambiguous: bool = False  # brand left out and another brand sells the same spec: only a human can pick
    anomaly: str | None = None


class TruthOrder(BaseModel):
    order_id: str
    customer_id: str
    sender: str
    po_number: str
    order_date: str
    delivery_date: str | None
    lines: list[TruthLine]
    anomaly: str | None
    format: str
    with_prices: bool


CUSTOMER_SEEDS = [
    ("Cabinet Dupont Avocats", "dupont-avocats.fr", "Claire Dupont", "fr", ["email", "email", "pdf"], "BC-{y}-{n:04d}"),
    ("Clinique des Oliviers", "clinique-oliviers.fr", "Karim Haddad", "fr", ["pdf", "photo"], "CO{n:06d}"),
    ("Hotel Atlas Marina", "atlas-marina-hotel.ma", "Nadia Bennani", "fr", ["email", "photo"], "HAM/{y}/{n:03d}"),
    ("Ecole Les Tilleuls", "ecole-tilleuls.fr", "Sophie Martin", "fr", ["email", "photo"], "ET-{n:03d}"),
    ("Transports Benali SARL", "benali-transports.ma", "Youssef Benali", "fr", ["email", "pdf"], "TB-{y}{n:04d}"),
    ("Pharmacie Centrale", "pharmacie-centrale.be", "Luc Peeters", "fr", ["email"], "PC{n:05d}"),
    ("Brightside Dental Group", "brightsidedental.co.uk", "Emma Clarke", "en", ["pdf", "email"], "PO-{y}-{n:05d}"),
    ("Northgate Academy", "northgate-academy.org.uk", "James Walker", "en", ["email", "pdf"], "NGA{n:05d}"),
    ("Lumen Architects Ltd", "lumen-architects.co.uk", "Priya Shah", "en", ["email"], "LA-{n:04d}"),
    ("Riverside Coworking", "riverside-cowork.ie", "Sean Murphy", "en", ["email", "photo"], "RC-{y}-{n:03d}"),
    ("Kestrel Engineering", "kestrel-eng.com", "Daniel Moore", "en", ["pdf"], "45000{n:05d}"),
    ("Harbour View Care Home", "harbourviewcare.co.uk", "Grace Evans", "en", ["email", "pdf"], "HV{n:04d}"),
]

USUAL_QTY = {
    "paper": (5, 40), "writing": (1, 5), "filing": (5, 30), "toner": (1, 4), "envelopes": (1, 5),
    "stapling": (1, 8), "notes": (1, 6), "tape": (1, 6), "notebooks": (10, 80), "labels": (1, 5),
    "batteries": (1, 10), "it": (1, 6), "bags": (5, 40), "gloves": (2, 30), "hygiene": (2, 20),
    "cleaning": (1, 12), "breakroom": (1, 20), "desk": (1, 10),
}

NOT_IN_CATALOG = {
    "en": ["Office chair mesh black", "Magnetic whiteboard 120x90cm", "HP LaserJet Pro M404dn printer",
           "Copy paper A4 160g white ream", "Nitrile gloves blue size XXL box 100", "Bin bags 240L black",
           "HP 30A toner black", "Coffee machine descaler 1L", "Laminating pouches A4 125 micron, box of 100",
           "Bic Cristal purple, box of 50", "Paper shredder cross-cut 12 sheets", "LED desk lamp"],
    "fr": ["Chaise de bureau mesh noire", "Tableau blanc magnetique 120x90cm", "Imprimante HP LaserJet Pro M404dn",
           "Ramette papier A4 160g blanc", "Gants nitrile bleu taille XXL boite de 100", "Sacs poubelle 240L noir",
           "Toner HP 30A noir", "Detartrant machine a cafe 1L", "Pochettes de plastification A4 125 microns, boite de 100",
           "Bic Cristal violet, boite de 50", "Destructeur de documents coupe croisee 12 feuilles", "Lampe de bureau LED"],
}

# (pattern, dev wordings, held-out wordings). Applied in order, first occurrence only, case-insensitive.
SUBS = {
    "en": [
        (r"coloured paper", ["coloured paper", "color paper"], ["tinted paper"]),
        (r"copy paper", ["copy paper", "printer paper", "paper"], ["copier paper", "multipurpose paper", "office paper"]),
        (r"ream of 500 sheets", ["ream of 500 sheets", "ream", "500 sheets", ""], ["500 sht ream", "per ream", ""]),
        (r"ballpoint pen medium", ["ballpoint pen", "ball pen", "pen medium"], ["biro", "ballpen M"]),
        (r"ballpoint pen fine", ["ballpoint fine", "pen fine"], ["biro fine", "fine ballpen"]),
        (r"ballpoint pen retractable", ["retractable pen", "ballpoint retractable"], ["click pen"]),
        (r"ballpoint pen extra broad", ["ballpoint XB", "pen extra broad"], ["biro extra bold"]),
        (r"gel pen 0\.7", ["gel pen", "gel pen 0.7mm"], ["gel rollerball 0.7"]),
        (r"whiteboard marker round tip", ["whiteboard marker", "wb marker"], ["dry wipe marker", "dry-erase marker"]),
        (r"permanent marker fine", ["permanent marker", "perm marker fine"], ["marker pen permanent"]),
        (r"highlighter", ["highlighter", "highlighters"], ["hi-liter", "text marker"]),
        (r"lever arch file", ["lever arch file", "lever arch", "arch file"], ["arch lever file", "lever-arch binder"]),
        (r"original toner cartridge", ["toner", "toner cartridge", "original toner"], ["laser toner", "print cartridge"]),
        (r"self-seal", ["self-seal", "self seal"], ["self adhesive"]),
        (r"peel and seal", ["peel and seal", "peel & seal"], ["strip seal"]),
        (r"no window", ["no window", "plain"], ["without window", "non window"]),
        (r"envelopes", ["envelopes", "env"], ["mailing envelopes"]),
        (r"galvanised", ["galvanised", ""], ["galv", ""]),
        (r"stapler", ["stapler"], ["stapling machine"]),
        (r"post-it notes", ["Post-it notes", "post-it", "sticky notes"], ["memo pads", "self-stick notes"]),
        (r"pack of 12 pads", ["pack of 12", "12 pads", ""], ["12/pack"]),
        (r"spiral notebook", ["spiral notebook", "notebook", "spiral pad"], ["wirebound notebook", "wire notebook"]),
        (r"laser labels", ["labels", "laser labels"], ["address labels", "sticky labels"]),
        (r"alkaline batteries", ["batteries", "alkaline batteries", "battery"], ["alkaline cells", "batts"]),
        (r"usb 3\.0 flash drive", ["flash drive", "usb stick", "usb key"], ["memory stick", "pen drive", "thumb drive"]),
        (r"bin bags", ["bin bags", "garbage bags", "trash bags"], ["bin liners", "refuse sacks", "rubbish bags"]),
        (r"roll of (\d+)", [r"roll of \1", r"\1/roll", r"x\1"], [r"\1 per roll"]),
        (r"disposable ", ["", "disposable "], [""]),
        (r"powder-free", ["powder-free", "powder free", "PF"], ["unpowdered", "non-powdered"]),
        (r"toilet paper", ["toilet paper", "toilet rolls"], ["loo rolls", "toilet tissue"]),
        (r"kitchen towel rolls", ["kitchen rolls", "kitchen towel"], ["paper towel rolls"]),
        (r"z-fold hand towels", ["z-fold towels", "hand towels z fold"], ["zfold paper towels"]),
        (r"liquid hand soap", ["hand soap", "liquid soap"], ["handwash"]),
        (r"dishwashing liquid", ["dish soap", "washing up liquid"], ["dishwash liquid"]),
        (r"multi-surface cleaner", ["multi surface cleaner", "all purpose cleaner"], ["multipurpose cleaner"]),
        (r"hand sanitiser gel", ["hand sanitiser", "sanitizer gel"], ["alcohol hand gel"]),
        (r"microfibre cloths", ["microfibre cloths", "microfiber cloths"], ["micro fibre wipes"]),
        (r"coffee beans", ["coffee beans", "bean coffee"], ["whole bean coffee"]),
        (r"paper cups", ["paper cups", "cups paper"], ["paper beakers"]),
        (r"wireless mouse", ["wireless mouse", "mouse"], ["cordless mouse"]),
        (r"box of (\d+)", [r"box of \1", r"bx \1", r"\1/box", r"x\1"], [r"\1 per box", r"(\1)"]),
        (r"pack of (\d+)", [r"pack of \1", r"pk \1", r"x\1"], [r"\1 per pack", r"\1-pack"]),
        (r"case of (\d+)", [r"case of \1", r"ctn \1"], [r"\1 per case"]),
        (r"original ", ["", "original "], [""]),
    ],
    "fr": [
        (r"ramette papier couleur", ["papier couleur", "ramette couleur"], ["papier teinte"]),
        (r"ramette papier", ["ramette papier", "ramettes", "rame papier", "papier"], ["papier reprographie", "papier photocopie", "ramette reprographique"]),
        (r"500 feuilles", ["500 feuilles", ""], ["500 ff", ""]),
        (r"stylo bille pointe moyenne", ["stylo bille", "stylos bille", "bille moyen"], ["pointe bille M", "stylo a bille"]),
        (r"stylo bille pointe fine", ["stylo bille fin", "stylos fins"], ["stylo a bille fine"]),
        (r"stylo bille retractable", ["stylo retractable", "bille retractable"], ["stylo a poussoir"]),
        (r"stylo bille extra large", ["stylo XB", "bille extra large"], ["stylo bille tres epais"]),
        (r"stylo gel 0\.7", ["stylo gel", "roller gel"], ["gel 0.7mm"]),
        (r"marqueur tableau blanc pointe ogive", ["marqueur tableau blanc", "marqueur effacable"], ["feutre tableau blanc", "marqueur velleda"]),
        (r"marqueur permanent pointe fine", ["marqueur permanent", "feutre permanent"], ["marqueur indelebile"]),
        (r"surligneur", ["surligneur", "surligneurs", "fluo"], ["stabilo fluo", "marqueur fluo"]),
        (r"classeur a levier", ["classeur a levier", "classeur levier", "classeur"], ["classeur a archives", "classeur levier carton"]),
        (r"toner (hp|brother|canon)", [r"toner \1", r"cartouche toner \1", r"cartouche \1"], [r"cartouche laser \1", r"toner laser \1"]),
        (r"avec fenetre", ["avec fenetre", "fenetre"], ["a fenetre"]),
        (r"sans fenetre", ["sans fenetre"], ["pleine", "sans fenetre"]),
        (r"autocollante", ["autocollante", "autocoll"], ["auto-adhesive"]),
        (r"bande protectrice", ["bande protectrice", "bande detachable"], ["bande siliconee"]),
        (r"enveloppes", ["enveloppes", "env"], ["enveloppes courrier"]),
        (r"agrafeuse", ["agrafeuse"], ["agrafeuse bureau"]),
        (r"notes post-it", ["post-it", "notes post-it", "blocs post-it"], ["notes repositionnables", "blocs repositionnables"]),
        (r"lot de 12 blocs", ["lot de 12", "12 blocs", ""], ["paquet de 12"]),
        (r"cahier spirale", ["cahier spirale", "cahier", "cahier spirales"], ["cahier reliure integrale", "bloc spirale"]),
        (r"petits carreaux 5x5", ["petits carreaux", "5x5", "quadrille 5x5"], ["carreaux 5x5"]),
        (r"grands carreaux seyes", ["seyes", "grands carreaux"], ["seyes grands carreaux"]),
        (r"etiquettes laser", ["etiquettes", "etiquettes laser"], ["planches etiquettes"]),
        (r"piles alcalines", ["piles", "piles alcalines"], ["piles alcaline"]),
        (r"\bAAA\b", ["AAA"], ["LR03"]),
        (r"\bAA\b", ["AA"], ["LR6"]),
        (r"cle usb 3\.0", ["cle usb", "cle usb 3.0"], ["memoire usb", "cle"]),
        (r"sacs poubelle", ["sacs poubelle", "sac poubelle", "sacs"], ["sacs a dechets", "sacs ordures", "sacs-poubelle"]),
        (r"rouleau de (\d+)", [r"rouleau de \1", r"x\1"], [r"\1 par rouleau"]),
        (r"gants jetables", ["gants", "gants jetables"], ["gants usage unique"]),
        (r"non poudres", ["non poudres", "sans poudre", ""], ["non talques"]),
        (r"boite de (\d+)", [r"boite de \1", r"bte \1", r"bte de \1", r"x\1"], [r"\1 par boite", r"(\1)"]),
        (r"lot de (\d+)", [r"lot de \1", r"x\1", r"pack de \1"], [r"\1 par lot", r"paquet de \1"]),
        (r"carton de (\d+)", [r"carton de \1", r"ctn \1"], [r"\1 par carton"]),
        (r"papier toilette", ["papier toilette", "papier wc"], ["papier hygienique"]),
        (r"essuie-tout", ["essuie-tout", "essuie tout"], ["sopalin"]),
        (r"essuie-mains pliage z", ["essuie-mains z", "essuie mains pliage z"], ["essuie-mains enchevetres"]),
        (r"savon liquide mains", ["savon liquide", "savon mains"], ["creme lavante mains"]),
        (r"liquide vaisselle", ["liquide vaisselle", "produit vaisselle"], ["detergent vaisselle"]),
        (r"nettoyant multi-surfaces", ["nettoyant multi surfaces", "multi-usages"], ["nettoyant toutes surfaces"]),
        (r"eau de javel", ["javel", "eau de javel"], ["chlore javel"]),
        (r"gel hydroalcoolique", ["gel hydroalcoolique", "gel hydro"], ["solution hydroalcoolique"]),
        (r"cafe en grains", ["cafe grains", "cafe en grain"], ["grains de cafe"]),
        (r"gobelets carton", ["gobelets carton", "gobelets papier"], ["tasses carton"]),
        (r" original ", [" ", " original "], [" "]),
    ],
}

BRAND_DROP = 0.4
TYPO = 0.12
FILLER = re.compile(r"\b(?:jetables|galvanisees|original)\b", re.IGNORECASE)


def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) >= 5]
    if not idx:
        return text
    i = rng.choice(idx)
    w, k = words[i], rng.randint(1, len(words[i]) - 2)
    words[i] = rng.choice([w[:k] + w[k + 1] + w[k] + w[k + 2:], w[:k] + w[k + 1:], w[:k] + w[k] + w[k:]])
    return " ".join(words)


def _format_attrs(s: str, p: Product, lang: str, pool: str, rng: random.Random) -> str:
    held = pool == "heldout"
    if p.category == "paper":
        s = re.sub(r"\b(\d{2,3})g\b", lambda m: rng.choice(
            [f"{m[1]} g/m2", f"{m[1]}grs", f"{m[1]} grammes"] if held else [f"{m[1]}g", f"{m[1]} g", f"{m[1]}gsm", f"{m[1]}gr"]), s)
    s = re.sub(r"\b(\d+)L\b", lambda m: rng.choice(
        [f"{m[1]} lt", f"{m[1]}-litre"] if held else [f"{m[1]}L", f"{m[1]} L", f"{m[1]} litres", f"{m[1]}ltr"]), s)
    if p.category == "filing":
        s = re.sub(r"\b(\d0)mm\b", lambda m: rng.choice(
            [f"{int(m[1]) // 10}cm", f"{int(m[1]) // 10} cm"] if held else [f"{m[1]}mm", f"{m[1]} mm"]), s)
    if p.category == "gloves":
        words = {"s": "small", "m": "medium", "l": "large", "xl": "XL"}

        def size(m: re.Match) -> str:
            v = m[2]
            return rng.choice([words[v.lower()], f"sz {v}"] if held else [f"{m[1]} {v}", v])
        s = re.sub(r"\b(size|taille) (XL|S|M|L)\b", size, s)
    if p.category == "toner" and p.brand == "HP":
        s = re.sub(r"\bHP (\d{2,3})([AX])\b", lambda m: rng.choice(
            [f"HP {m[1]} {m[2]}", f"HP{m[1]}{m[2]}"] if held else [f"HP {m[1]}{m[2]}", f"{m[1]}{m[2]}"]), s)
    if lang == "en" and rng.random() < 0.3:
        s = re.sub(r"\b(black|white|blue)\b", lambda m: {"black": "blk", "white": "wht", "blue": "blu"}[m[1]], s)
    if lang == "fr" and held and rng.random() < 0.5:
        s = re.sub(r"\b(bleu|noir|blanc)\b", lambda m: rng.choice({"bleu": ["bleue", "bleus"], "noir": ["noire", "noirs"],
                                                                     "blanc": ["blanche", "blanches"]}[m[1]]), s)
    return s


def phrase(p: Product, lang: str, pool: str, rng: random.Random) -> str:
    """How a customer might write this product."""
    s = p.name_en if lang == "en" else p.name_fr
    if rng.random() < 0.8:
        s = re.sub(r"\s*\([^)]*\)", "", s)  # drop the "(CF226A)" part
    for pattern, dev, held in SUBS[lang]:
        if re.search(pattern, s, re.IGNORECASE):
            choices = held if pool == "heldout" and rng.random() < 0.7 else dev
            s = re.sub(pattern, rng.choice(choices), s, count=1, flags=re.IGNORECASE)
    s = _format_attrs(s, p, lang, pool, rng)
    if p.brand not in ("Generic", "Pro Clean") and rng.random() < BRAND_DROP:
        s = re.sub(rf"\b{re.escape(p.brand)}\b", "", s, flags=re.IGNORECASE)
    s = FILLER.sub("", s)
    s = re.sub(r"\s+,", ",", re.sub(r"\s{2,}", " ", s)).strip(" ,")
    if rng.random() < TYPO:
        s = _typo(s, rng)
    r = rng.random()
    return s.lower() if r < 0.35 else s.upper() if r < 0.42 else s


def make_customers(catalog: list[Product], seed: int = 3) -> list[Customer]:
    rng = random.Random(seed)
    active = [p for p in catalog if p.status == "active"]
    customers = []
    for i, (name, domain, contact, lang, formats, po_fmt) in enumerate(CUSTOMER_SEEDS, 1):
        c = Customer(id=f"C{i:03d}", name=name, domain=domain, contact=contact, lang=lang,
                     discount=rng.choice([0, 0.05, 0.08, 0.1, 0.12]), own_codes=i in (2, 7, 11),
                     knows_skus=i in (5, 8, 12), formats=formats, po_format=po_fmt)
        for p in rng.sample(active, rng.randint(12, 22)):
            c.habitual[p.sku] = phrase(p, lang, "dev", rng)
            lo, hi = USUAL_QTY[p.category]
            c.usual_qty[p.sku] = rng.randint(lo, hi)
            if c.own_codes:
                c.own_code_map[p.sku] = f"{rng.choice(['ART', 'MAT', 'FN'])}-{rng.randint(1000, 9999)}"
        customers.append(c)
    return customers


def customer_price(p: Product, c: Customer) -> float:
    return round(p.price * (1 - c.discount), 2)


def make_history(customers: list[Customer], n_past: int = 6, seed: int = 5) -> dict[str, dict[str, list[float]]]:
    """Past ordered quantities per customer and product, used for the unusual-quantity check."""
    rng = random.Random(seed)
    return {c.id: {sku: [max(1, round(q * rng.choice([0.5, 0.75, 1, 1, 1.25, 1.5, 2]))) for _ in range(n_past)]
                   for sku, q in c.usual_qty.items()} for c in customers}


def _po(c: Customer, rng: random.Random) -> str:
    return c.po_format.format(y=2026, n=rng.randint(1, 9999 if "{n:04d}" in c.po_format or "05d" in c.po_format else 999))


def generate_orders(catalog: list[Product], customers: list[Customer], n: int, pool: str, seed: int,
                    anomaly_share: float = 0.4, start: date = date(2026, 9, 1)) -> list[TruthOrder]:
    rng = random.Random(seed)
    by_sku = {p.sku: p for p in catalog}
    active = [p for p in catalog if p.status == "active"]
    discontinued = [p for p in catalog if p.status == "discontinued"]
    n_anom = round(n * anomaly_share)
    plan = [ANOMALIES[i % len(ANOMALIES)] for i in range(n_anom)] + [None] * (n - n_anom)
    rng.shuffle(plan)
    orders: list[TruthOrder] = []

    for i, anomaly in enumerate(plan, 1):
        c = rng.choice(customers)
        fmt = rng.choice(c.formats)
        with_prices = fmt != "email" and (c.own_codes or rng.random() < 0.5)
        if anomaly == "price_mismatch":
            with_prices = True
            if fmt == "email":
                fmt = "pdf"
        if anomaly == "duplicate_po" and not any(o.customer_id == c.id for o in orders):
            anomaly = None
        n_lines = rng.randint(2, 8)
        lines: list[TruthLine] = []
        skus_used: set[str] = set()
        habitual = list(c.habitual)
        for _ in range(n_lines):
            r = rng.random()
            if r < 0.6 and habitual:
                sku = rng.choice(habitual)
                if sku in skus_used:
                    continue
                p = by_sku[sku]
                text, kind = c.habitual[sku], "repeat"
                qty = max(1, round(c.usual_qty[sku] * rng.choice([0.5, 1, 1, 1, 1.5, 2])))
            else:
                p = rng.choice(active)
                if p.sku in skus_used:
                    continue
                lo, hi = USUAL_QTY[p.category]
                qty = rng.randint(lo, hi)
                kind = "new"
                if c.knows_skus and rng.random() < 0.3:
                    text, kind = f"Ref {p.sku} {phrase(p, c.lang, pool, rng)[:28].strip()}", "sku_code"
                elif p.category in ("toner", "labels") and rng.random() < 0.4:
                    text, kind = f"{p.mpn} {phrase(p, c.lang, pool, rng)}", "mpn_code"
                else:
                    text = phrase(p, c.lang, pool, rng)
            skus_used.add(p.sku)
            ref = c.own_code_map.get(p.sku) if c.own_codes else None
            if c.own_codes and ref is None:
                ref = f"{rng.choice(['ART', 'MAT', 'FN'])}-{rng.randint(1000, 9999)}"  # new article in their ERP
            lines.append(TruthLine(text=text, customer_ref=ref, quantity=qty, sku=p.sku, kind=kind,
                                   unit_price=customer_price(p, c) if with_prices else None))

        if anomaly == "qty_outlier":
            reps = [ln for ln in lines if ln.kind == "repeat"]
            if reps:
                ln = rng.choice(reps)
                ln.quantity, ln.anomaly = max(ln.quantity, c.usual_qty[ln.sku]) * 10, "qty_outlier"
            else:
                anomaly = None
        elif anomaly == "price_mismatch":
            ln = rng.choice(lines)
            ln.unit_price, ln.anomaly = round(ln.unit_price * rng.choice([0.85, 0.9, 1.12, 1.2]), 2), "price_mismatch"
        elif anomaly == "discontinued":
            p = rng.choice(discontinued)
            lines.insert(rng.randint(0, len(lines)), TruthLine(
                text=phrase(p, c.lang, pool, rng), quantity=rng.randint(*USUAL_QTY[p.category]), sku=p.sku,
                kind="new", anomaly="discontinued", customer_ref=f"ART-{rng.randint(1000, 9999)}" if c.own_codes else None,
                unit_price=customer_price(p, c) if with_prices else None))
        elif anomaly == "not_in_catalog":
            text = rng.choice(NOT_IN_CATALOG[c.lang])
            lines.insert(rng.randint(0, len(lines)), TruthLine(
                text=text, quantity=rng.randint(1, 5), sku=None, kind="not_in_catalog", anomaly="not_in_catalog",
                customer_ref=f"ART-{rng.randint(1000, 9999)}" if c.own_codes else None,
                unit_price=round(rng.uniform(5, 300), 2) if with_prices else None))

        order_date = start + timedelta(days=rng.randint(0, 40))
        po = _po(c, rng)
        if anomaly == "duplicate_po":
            po = rng.choice([o for o in orders if o.customer_id == c.id]).po_number
        delivery = (order_date + timedelta(days=rng.randint(2, 10))).isoformat() if rng.random() < 0.6 else None
        first = c.contact.split()[0].lower()
        orders.append(TruthOrder(
            order_id=f"ord_{i:03d}", customer_id=c.id, sender=f"{first}@{c.domain}", po_number=po,
            order_date=order_date.isoformat(), delivery_date=delivery, lines=lines, anomaly=anomaly, format=fmt,
            with_prices=with_prices))
    for o in orders:
        for ln in o.lines:
            ln.ambiguous = ln.kind in ("repeat", "new") and ln.sku is not None and is_ambiguous(ln.text, by_sku[ln.sku], catalog)
    return orders


def is_ambiguous(text: str, p: Product, catalog: list[Product]) -> bool:
    """True when the brand is missing from the text and another brand sells a product with the same spec."""
    if p.brand.lower() in text.lower():
        return False
    return any(q.category == p.category and q.attrs == p.attrs and q.brand != p.brand for q in catalog)
