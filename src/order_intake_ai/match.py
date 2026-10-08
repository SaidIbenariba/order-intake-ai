"""Match an order line to a catalog product.

Cascade, stopping at the first stage that is sure:
1. code: the line quotes one of our SKUs or a manufacturer part number (CF226A, TN-2420, L7160)
2. memory: this customer used this exact article code or wording before and staff confirmed the product
3. hybrid score over the whole catalog: fuzzy text similarity + multilingual embeddings (EN/FR) +
   attribute agreement (size, grams, colour, litres, pack...). A contradicted attribute ("A4 160g" vs a
   100g product) cuts the score hard, because near-miss products are the costly mistakes.

A line is auto-matched only if the best score clears a threshold AND beats the runner-up by a margin.
Everything else goes to review with the top 3 candidates. Thresholds are tuned on the dev batch for
zero wrong auto-matches, then frozen.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from .catalog import Product
from .text import compact_code, compare_attrs, normalize, parse_attrs

Encoder = Callable[[list[str]], np.ndarray]  # rows L2-normalised
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

W_FUZZY, W_EMBED, W_ATTR = 0.35, 0.45, 0.20
CONFLICT_FACTOR = 0.35  # per contradicted attribute
MISSING_FACTOR = 0.9  # per attribute the line states but the product doesn't have


def default_encoder(model_name: str = EMBED_MODEL) -> Encoder:
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=64)


@dataclass
class Candidate:
    sku: str
    name: str
    score: float
    fuzzy: float = 0.0
    embed: float = 0.0
    conflicts: list[str] = field(default_factory=list)


@dataclass
class Match:
    status: str  # AUTO | REVIEW | NO_MATCH
    method: str  # code | memory | hybrid | none
    sku: str | None
    score: float
    margin: float
    candidates: list[Candidate]
    reason: str = ""


class Memory:
    """Confirmed (customer, article code or wording) -> SKU pairs. Grows every time staff approve a line."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.pairs: dict[str, str] = {}
        if path and path.exists():
            self.pairs = json.loads(path.read_text())

    @staticmethod
    def _keys(customer_id: str, text: str, customer_ref: str | None) -> list[str]:
        keys = [f"{customer_id}|ref|{compact_code(customer_ref)}"] if customer_ref else []
        return keys + [f"{customer_id}|txt|{normalize(text)}"]

    def get(self, customer_id: str, text: str, customer_ref: str | None) -> str | None:
        for k in self._keys(customer_id, text, customer_ref):
            if k in self.pairs:
                return self.pairs[k]
        return None

    def learn(self, customer_id: str, text: str, customer_ref: str | None, sku: str) -> None:
        for k in self._keys(customer_id, text, customer_ref):
            self.pairs[k] = sku

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.pairs, indent=1, ensure_ascii=False))


class Matcher:
    def __init__(self, catalog: list[Product], encoder: Encoder | None = None, tau: float = 0.62, margin: float = 0.06,
                 tau_none: float = 0.35):
        self.catalog = catalog
        self.by_sku = {p.sku: p for p in catalog}
        self.tau, self.margin, self.tau_none = tau, margin, tau_none
        self.codes: dict[str, str] = {}
        for p in catalog:
            self.codes[compact_code(p.sku)] = p.sku
            self.codes[compact_code(p.mpn)] = p.sku
        self.names = [(normalize(p.name_en), normalize(p.name_fr)) for p in catalog]
        self.brands = sorted({normalize(p.brand) for p in catalog if p.brand != "Generic"}, key=len, reverse=True)
        self.attrs = []
        for p in catalog:
            a = parse_attrs(p.name_en)
            if p.brand != "Generic":
                a["brand"] = {normalize(p.brand)}
            for k, v in parse_attrs(p.name_fr).items():
                a.setdefault(k, set()).update(v)
            self.attrs.append(a)
        self.encoder = encoder or default_encoder()
        vecs = self.encoder([n for pair in self.names for n in pair])
        self.vec_en, self.vec_fr = vecs[0::2], vecs[1::2]
        self._cache: dict[str, np.ndarray] = {}

    # ---- stages ----
    def find_code(self, text: str, customer_ref: str | None = None) -> str | None:
        tokens = [t for t in normalize(text).replace("/", " ").split()]
        if customer_ref:  # the model sometimes moves a code it sees ("AVERY L7160", "Ref 537416") into the ref field
            text = f"{customer_ref} {text}"
            tokens = normalize(text).replace("/", " ").split()
        raw = [t for t in text.replace(",", " ").replace("(", " ").replace(")", " ").split()]
        cands = {compact_code(t) for t in tokens + raw}
        cands |= {compact_code(a + b) for a, b in zip(raw, raw[1:])}  # "TN 2420", "CF 226A"
        if customer_ref:
            cands.add(compact_code(customer_ref))
        hits = {self.codes[c] for c in cands if len(c) >= 4 and c in self.codes}
        return hits.pop() if len(hits) == 1 else None

    def embed(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            for t, v in zip(new, self.encoder(new)):
                self._cache[t] = np.asarray(v)
        return np.stack([self._cache[t] for t in texts])

    def line_attrs(self, text: str) -> dict[str, set[str]]:
        a = parse_attrs(text)
        norm = f" {normalize(text)} "
        found = {b for b in self.brands if f" {b} " in norm}
        found = {b for b in found if not any(b != o and b in o for o in found)}  # "clairefontaine" inside "clairefontaine trophee"
        if found:
            a["brand"] = found
        return a

    def score_all(self, text: str) -> list[Candidate]:
        norm = normalize(text)
        q = self.embed([norm])[0]
        emb = np.maximum(self.vec_en @ q, self.vec_fr @ q)
        line_attrs = self.line_attrs(text)
        out = []
        for i, p in enumerate(self.catalog):
            en, fr = self.names[i]
            fz = max(fuzz.token_set_ratio(norm, en), fuzz.token_set_ratio(norm, fr),
                     fuzz.token_sort_ratio(norm, en), fuzz.token_sort_ratio(norm, fr)) / 100
            agree, conflict, missing = compare_attrs(line_attrs, self.attrs[i])
            stated = agree + conflict + missing
            attr = agree / stated if stated else 0.5
            s = (W_FUZZY * fz + W_EMBED * float(emb[i]) + W_ATTR * attr) * CONFLICT_FACTOR ** conflict * MISSING_FACTOR ** missing
            conflicts = [k for k, v in line_attrs.items() if k in self.attrs[i] and not (v & self.attrs[i][k])]
            out.append(Candidate(p.sku, p.name_en, round(s, 4), round(fz, 3), round(float(emb[i]), 3), conflicts))
        out.sort(key=lambda c: -c.score)
        return out

    # ---- decision ----
    def match(self, text: str, customer_id: str | None = None, customer_ref: str | None = None,
              memory: Memory | None = None) -> Match:
        sku = self.find_code(text, customer_ref)
        if sku:
            p = self.by_sku[sku]
            return Match("AUTO", "code", sku, 1.0, 1.0, [Candidate(sku, p.name_en, 1.0)], "product code in the line")
        if memory and customer_id:
            sku = memory.get(customer_id, text, customer_ref)
            if sku and sku in self.by_sku:
                p = self.by_sku[sku]
                return Match("AUTO", "memory", sku, 1.0, 1.0, [Candidate(sku, p.name_en, 1.0)],
                             "same wording confirmed for this customer before")
        cands = self.score_all(text)
        top, second = cands[0], cands[1]
        m = top.score - second.score
        if top.score < self.tau_none:
            return Match("NO_MATCH", "none", None, top.score, m, cands[:3], "no product close enough: not in catalog?")
        if top.score >= self.tau and m >= self.margin:
            return Match("AUTO", "hybrid", top.sku, top.score, m, cands[:3], "clear best match")
        why = "low score" if top.score < self.tau else f"close alternatives ({second.name})"
        return Match("REVIEW", "hybrid", None, top.score, m, cands[:3], why)
