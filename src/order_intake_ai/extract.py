"""Read an order email (.eml) and extract the order with a local model on Ollama. Nothing leaves the machine.

Source routing:
- PDF purchase order with a text layer -> text path (model reads the exact text)
- photo / scan attachment               -> vision path (model reads the image)
- no attachment                         -> the email body is the order

The model only COPIES strings. Numbers and dates are normalized by deterministic code.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

import httpx
import pdfplumber
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl"
MAX_WIDTH = 1000

_S = {"type": "string"}
RAW_SCHEMA = {
    "type": "object",
    "properties": {
        "po_number": _S, "order_date": _S, "delivery_date": _S,
        "lines": {"type": "array", "items": {
            "type": "object",
            "properties": {"customer_ref": _S, "description": _S, "quantity": _S, "unit_price": _S},
            "required": ["customer_ref", "description", "quantity", "unit_price"],
        }},
    },
    "required": ["po_number", "order_date", "delivery_date", "lines"],
}

PROMPT = """This is a purchase order sent by a customer to a supplier. Copy the values below EXACTLY as written (same words, same spelling mistakes, same digits). Do not translate, correct, complete or recompute anything. Use "" when a value is not written.

- po_number: the customer's order number (PO, PO Number, Bon de commande n°, N° commande), value only
- order_date: the order date as written ("" if only in the email header)
- delivery_date: the requested delivery date as written, only if a delivery date is written ("" otherwise; never reuse the order date)
- lines: one entry per ordered product, in order:
  - customer_ref: the customer's own article code if there is a Ref column or a code before the description (like ART-1234), else ""
  - description: the product description exactly as written, including sizes, colours and pack sizes (box of 50, x100, bte 50...)
  - quantity: the number of units ordered (the Qty / Qte column, or the number before "x", or after "qty"/":")
  - unit_price: the unit price if written, else \"\""""


@dataclass
class EmailOrder:
    sender: str
    subject: str
    sent: str | None  # ISO date from the email header
    body: str
    attachment_name: str | None = None
    attachment: bytes | None = None
    attachment_type: str | None = None  # "pdf" | "image"


@dataclass
class ExtractedLine:
    customer_ref: str | None
    description: str
    quantity: float | None
    unit_price: float | None
    qty_note: str | None = None  # set when the written line suggests another quantity


@dataclass
class ExtractedOrder:
    sender: str
    po_number: str | None
    order_date: str | None
    delivery_date: str | None
    lines: list[ExtractedLine]
    route: str
    seconds: float
    raw: dict = field(default_factory=dict)


def read_eml(path: Path) -> EmailOrder:
    msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = body_part.get_content() if body_part else ""
    sent = None
    try:
        sent = parsedate_to_datetime(msg["Date"]).date().isoformat()
    except (TypeError, ValueError):
        pass
    order = EmailOrder(sender=parseaddr(msg["From"] or "")[1].lower(), subject=msg["Subject"] or "", sent=sent, body=body)
    for part in msg.iter_attachments():
        ctype = part.get_content_type()
        if ctype == "application/pdf" or ctype.startswith("image/"):
            order.attachment_name = part.get_filename()
            order.attachment = part.get_content()
            order.attachment_type = "pdf" if ctype == "application/pdf" else "image"
            break
    return order


def pdf_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:  # layout mode keeps table columns aligned per row
        text = "\n".join((p.extract_text(layout=True) or "") for p in pdf.pages)
    return "\n".join(ln.rstrip() for ln in text.splitlines() if ln.strip())


def _b64_image(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------- deterministic normalization ----------

def parse_number(raw: str | None) -> float | None:
    if not raw:
        return None
    s = re.sub(r"[^\d,.\-]", "", raw.replace("\xa0", "").replace(" ", ""))
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def parse_date(raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    cands = []
    if m := re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        cands.append((int(m[1]), int(m[2]), int(m[3])))
    if m := re.search(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", s):  # day first (FR / UK / MA)
        cands.append((int(m[3]), int(m[2]), int(m[1])))
    if (m := re.search(r"(\d{1,2})\s+([A-Za-z]{3})[a-z]*\.?\s+(\d{4})", s)) and m[2].lower() in _MONTHS:
        cands.append((int(m[3]), _MONTHS[m[2].lower()], int(m[1])))
    for y, mo, d in cands:
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            continue
    return None


_PO_LABEL = re.compile(r"^\s*(?:po|p\.o\.|purchase order|bon de commande|commande|order)?\s*(?:number|no\.?|n°|nº|#)?\s*[:#]?\s*",
                       re.IGNORECASE)


def clean_po(raw: str | None) -> str | None:
    """Drop a label copied with the value ("PO: LA-9756" -> "LA-9756") but keep "PO-2026-03080" whole."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    m = _PO_LABEL.match(s)
    if m and m.end() > 0 and s[m.end() - 1] in " :#°\t":
        s = s[m.end():].strip()
    return s or raw.strip()


_LEADING_NUM = re.compile(r"^(\d+(?:[.,]\d+)?)\s+(?:x\s+)?(?=\S)", re.IGNORECASE)
_TRAILING_NUM = re.compile(r"\s+(\d+(?:[.,]\d+)?)$")


def clean_line(desc: str, qty: float | None) -> tuple[str, str | None]:
    """Remove a quantity the model copied into the description; flag a leading number that disagrees with it.

    "10 Pencil HB, x12" with quantity 10 -> "Pencil HB, x12".
    "1 marker fine blue, 12 par boite" with quantity 12 -> kept, flagged: the line starts with 1.
    """
    note = None
    if m := _LEADING_NUM.match(desc):
        n = parse_number(m[1])
        if qty is not None and n == qty:
            desc = desc[m.end():]
        else:
            note = f"Line starts with {m[1]} but quantity was read as {qty:g}" if qty is not None else None
    elif (m := _TRAILING_NUM.search(desc)) and qty is not None and parse_number(m[1]) == qty:
        desc = desc[:m.start()]
    return desc.strip(), note


def _grounded(value: str | None, source: str | None) -> str | None:
    """Keep a copied value only if it appears in the source text (text routes). Models sometimes fill a field
    the document doesn't have, e.g. reuse the order date as the delivery date."""
    if not value or not value.strip():
        return None
    if source is None:  # vision route: nothing to check against
        return value
    squash = lambda t: re.sub(r"\s+", "", t).lower()
    return value if squash(value) in squash(source) else None


_DELIVERY_LABEL = re.compile(r"deliver|livraison|livrer|livre|ship|due|souhaitee", re.IGNORECASE)


def fold_text(t: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def source_text(mail: EmailOrder) -> tuple[str, str]:
    """(route, text the model reads). Text is empty for the vision route."""
    header = f"Email subject: {mail.subject}\nEmail body:\n{mail.body}"
    if mail.attachment_type == "pdf" and len(text := pdf_text(mail.attachment)) > 80:
        return "pdf-text", f"{header}\n\nAttached purchase order (text):\n{text}"
    if mail.attachment_type in ("pdf", "image"):
        return "vision", header
    return "email-body", header


def to_order(raw: dict, mail: EmailOrder, route: str, seconds: float, source: str | None = None) -> ExtractedOrder:
    """Model output (raw strings) -> clean order. `source` is the text the model read, for grounding checks."""
    if route == "vision":
        source = None
    lines = []
    for li in raw.get("lines") or []:
        desc = (li.get("description") or "").strip()
        if not desc:
            continue
        qty = parse_number(li.get("quantity"))
        desc, note = clean_line(desc, qty)
        lines.append(ExtractedLine(customer_ref=(li.get("customer_ref") or "").strip() or None, description=desc,
                                   quantity=qty, unit_price=parse_number(li.get("unit_price")), qty_note=note))
    order_raw, delivery_raw = _grounded(raw.get("order_date"), source), _grounded(raw.get("delivery_date"), source)
    if source is not None and delivery_raw and not _DELIVERY_LABEL.search(fold_text(source)):
        # no delivery wording anywhere: the model put the document's only date in the wrong field
        order_raw, delivery_raw = order_raw or delivery_raw, None
    return ExtractedOrder(sender=mail.sender, po_number=clean_po(raw.get("po_number")),
                          order_date=parse_date(order_raw) or mail.sent, delivery_date=parse_date(delivery_raw),
                          lines=lines, route=route, seconds=seconds, raw=raw)


def extract(path: Path, model: str = DEFAULT_MODEL, timeout: float = 900) -> ExtractedOrder:
    mail = read_eml(path)
    route, text = source_text(mail)
    if route == "vision":
        data = mail.attachment
        if mail.attachment_type == "pdf":
            import pypdfium2 as pdfium
            page = pdfium.PdfDocument(data)[0]
            buf = io.BytesIO()
            page.render(scale=MAX_WIDTH / page.get_width()).to_pil().save(buf, format="PNG")
            data = buf.getvalue()
        msg = {"role": "user", "content": f"{PROMPT}\n\n{text}\n\nThe purchase order is in the attached image.",
               "images": [_b64_image(data)]}
    else:
        msg = {"role": "user", "content": f"{PROMPT}\n\n{text}"}
    body = {"model": model, "messages": [msg], "format": RAW_SCHEMA, "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192}}
    t0 = time.perf_counter()
    r = httpx.post(OLLAMA_URL, json=body, timeout=timeout)
    r.raise_for_status()
    raw = json.loads(r.json()["message"]["content"])
    return to_order(raw, mail, route, time.perf_counter() - t0, text)
