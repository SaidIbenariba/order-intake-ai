"""Text normalization and attribute parsing for product descriptions (EN/FR).

The same parser runs on catalog names and on customer order lines, so attributes are compared like
for like: "A4 80gsm blk" and "Ramette A4 80 g/m2 noir" both give {size: a4, grams: 80, color: black}.
"""

from __future__ import annotations

import re
import unicodedata

# A few dozen trade abbreviations a sales-admin team uses every day. Kept short on purpose.
ABBREV = {
    "blk": "black", "bk": "black", "blu": "blue", "wht": "white", "grn": "green", "rd": "red",
    "yel": "yellow", "ylw": "yellow", "pk": "pack", "pkt": "pack", "pck": "pack", "bx": "box",
    "bte": "boite", "bt": "boite", "ctn": "carton", "env": "envelope", "envs": "envelopes",
    "tnr": "toner", "cart": "cartridge", "rm": "ream", "qty": "", "pcs": "pieces", "pc": "pieces",
    "nb": "", "ref": "", "art": "",
}

COLOR_FORMS = {
    "blue": ["blue", "blu", "bleu", "bleue", "bleus", "bleues"],
    "black": ["black", "blk", "bk", "noir", "noire", "noirs", "noires"],
    "red": ["red", "rouge", "rouges"],
    "green": ["green", "grn", "vert", "verte", "verts", "vertes"],
    "yellow": ["yellow", "yel", "ylw", "jaune", "jaunes"],
    "pink": ["pink", "rose", "roses"],
    "orange": ["orange", "oranges"],
    "white": ["white", "wht", "blanc", "blanche", "blancs", "blanches"],
    "ivory": ["ivory", "ivoire"],
    "grey": ["grey", "gray", "gris", "grise"],
    "purple": ["purple", "violet", "violette"],
    "transparent": ["transparent", "transparente", "transparents", "clear", "translucide"],
    "cyan": ["cyan"],
    "magenta": ["magenta"],
}
_COLOR = {form: c for c, forms in COLOR_FORMS.items() for form in forms}

_BATTERY = {"aa": "aa", "lr6": "aa", "aaa": "aaa", "lr03": "aaa", "9v": "9v", "6lr61": "9v",
            "lr14": "c", "lr20": "d"}
_GLOVE = {"xs": "xs", "s": "s", "m": "m", "l": "l", "xl": "xl", "xxl": "xxl", "small": "s", "medium": "m",
          "large": "l", "petit": "s", "moyen": "m", "grand": "l"}
_RULING = {"lined": "lined", "ruled": "lined", "ligne": "lined", "lignes": "lined", "squared": "squared",
           "quad": "squared", "quadrille": "squared", "seyes": "seyes"}
# words that look like attributes but are part of a product name ("tableau blanc" is a whiteboard, not white)
_NAME_PHRASES = r"\b(?:tableau blanc|carte noire|yellow label|grands carreaux|petits carreaux)\b"
_MATERIAL = {"nitrile": "nitrile", "latex": "latex", "vinyl": "vinyl", "vinyle": "vinyl"}
_PACK_WORDS = r"(?:pack|pk|pkt|box|bx|lot|boite|bte|bt|carton|ctn|case|roll|rouleau|paquet|ream|rame|ramette|bag|sachet)s?"
_COUNT_WORDS = r"(?:sheets|feuilles|formats|rolls|rouleaux|pieces|pcs|units|unites|sachets|pads|blocs|bags)"


def fold(text: object) -> str:
    """Lowercase and strip accents, keep punctuation."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()


def normalize(text: object) -> str:
    """Lowercase, strip accents, expand trade abbreviations, one space between tokens."""
    s = fold(text)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)  # 1,5l -> 1.5l
    tokens = re.sub(r"[^a-z0-9./]+", " ", s).split()
    out = []
    for t in tokens:
        t = t.strip("./")
        if not t:
            continue
        t = ABBREV.get(t, t)
        if t:
            out.append(t)
    return " ".join(out)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _fmt(x: float) -> str:
    return f"{x:g}"


def parse_attrs(text: object) -> dict[str, set[str]]:
    """Comparable attributes found in a product description. Values are sets (a text may hold several)."""
    s = fold(text)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    a: dict[str, set[str]] = {}

    def add(key: str, value: str) -> None:
        a.setdefault(key, set()).add(value)

    def take(pattern: str, fn) -> None:
        nonlocal s
        for m in re.finditer(pattern, s):
            fn(m)
        s = re.sub(pattern, " ", s)

    # codes and dimensions first, so their digits aren't read as weights or volumes
    take(r"\b(\d{2})\s*/\s*(\d{1,2})\b", lambda m: add("staple", f"{m[1]}/{m[2]}"))
    take(r"\b(\d+(?:\.\d+)?)\s*(?:mm|cm|m)?\s*x\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?\b",
         lambda m: add("dims", f"{_fmt(_num(m[1]))}x{_fmt(_num(m[2]))}"))
    take(r"\b(?:hp\s*)?(\d{2,3})\s?([ax])\b", lambda m: add("model", f"{m[1]}{m[2]}"))
    take(r"\btn\s*-?\s*(\d{3,4}(?:bk|c|m|y)?)\b", lambda m: add("model", f"tn-{m[1]}"))
    take(r"\b(?:crg\s*-?\s*)?(05[24]h?)\b", lambda m: add("model", m[1]))
    take(r"\b(l7\d{3})\b", lambda m: add("model", m[1]))
    take(r"\b(\d+)\s*(?:gb|go)\b", lambda m: add("capacity", f"{m[1]}gb"))
    take(r"\b(\d+(?:\.\d+)?)\s*(?:g/m2|g/m²|gsm|grammes|grams|grs|gr|g)\b", lambda m: add("grams", _fmt(_num(m[1]))))
    take(r"\b(\d+(?:\.\d+)?)\s*kg\b", lambda m: add("grams", _fmt(_num(m[1]) * 1000)))
    take(r"\b(\d+(?:\.\d+)?)[\s-]*(?:litres?|liters?|ltrs?|lt|l)\b", lambda m: add("litres", _fmt(_num(m[1]))))
    take(r"\b(\d+(?:\.\d+)?)\s*ml\b", lambda m: add("litres", _fmt(_num(m[1]) / 1000)))
    take(r"\b(\d+(?:\.\d+)?)\s*cl\b", lambda m: add("litres", _fmt(_num(m[1]) / 100)))
    take(r"\b(\d+(?:\.\d+)?)\s*mm\b", lambda m: add("mm", _fmt(_num(m[1]))))
    take(r"\b(\d+(?:\.\d+)?)\s*cm\b", lambda m: add("mm", _fmt(_num(m[1]) * 10)))
    take(r"\b(\d+)\s*(?:pages|pp)\b", lambda m: add("pages", m[1]))
    take(r"\b(\d+)\s*(?:plis|ply)\b|\b(\d+)-ply\b", lambda m: add("ply", m[1] or m[2]))
    take(rf"\b{_PACK_WORDS}\s*(?:of|de|x)?\s*(\d+)\b", lambda m: add("pack", m[1]))
    take(rf"\b(\d+)\s*{_COUNT_WORDS}\b", lambda m: add("pack", m[1]))
    take(r"\bx\s?(\d+)\b|\b(\d+)\s?/\s?(?:box|bte|pack)\b|\((\d+)\)", lambda m: add("pack", m[1] or m[2] or m[3]))

    if re.search(r"\bgrands carreaux\b", s):
        add("ruling", "seyes")
    if re.search(r"\bpetits carreaux\b", s):
        add("ruling", "squared")
    s = re.sub(_NAME_PHRASES, " ", s)
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    gloves = any(w.startswith(("glove", "gant")) for w in words)
    batteries = any(w.startswith(("batter", "pile")) for w in words)
    for i, w in enumerate(words):
        if w in ("a3", "a4", "a5"):
            add("size", w)
        elif w in ("dl", "c4", "c5", "c6"):
            add("format", w)
        elif w in _COLOR:
            add("color", _COLOR[w])
        elif w in _BATTERY:
            add("battery", _BATTERY[w])
        elif w in _RULING:
            add("ruling", _RULING[w])
        elif w in _MATERIAL:
            add("material", _MATERIAL[w])
        elif w in ("azerty", "qwerty"):
            add("layout", w)
        elif w in ("size", "taille", "sz", "t") and i + 1 < len(words) and words[i + 1] in _GLOVE:
            add("glove_size", _GLOVE[words[i + 1]])
        elif w in ("xs", "xl", "xxl", "medium", "small", "large") or (gloves and w in ("s", "m", "l")):
            add("glove_size", _GLOVE[w])
        elif batteries and w in ("c", "d"):
            add("battery", w)

    joined = " ".join(words)
    if re.search(r"\bself[\s-]?seal|\bautocoll", s):
        add("closure", "self")
    if re.search(r"\bpeel\s*(?:and|&)\s*seal|\bbande (?:protectrice|detachable)", s):
        add("closure", "peel")
    if re.search(r"\b(?:no|without|non|sans)\s+(?:window|fenetre)s?\b|\bplain\b", joined):
        add("window", "no")
    elif re.search(r"\b(?:window|fenetre)s?\b", joined):
        add("window", "yes")
    return a


def compare_attrs(line: dict[str, set[str]], product: dict[str, set[str]]) -> tuple[int, int, int]:
    """(agreements, conflicts, missing): keys the line states that the product agrees with, contradicts or lacks."""
    agree = conflict = missing = 0
    for key, values in line.items():
        if key not in product:
            missing += 1
        elif values & product[key]:
            agree += 1
        else:
            conflict += 1
    return agree, conflict, missing


def compact_code(text: str) -> str:
    """Code comparison key: uppercase, no separators ("tn 2420" -> "TN2420")."""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()
