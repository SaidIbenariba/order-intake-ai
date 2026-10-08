"""Turn truth orders into the files a sales-admin inbox actually receives: .eml emails with the order in
the body, or with a PDF purchase order attached, or with a phone photo of the purchase order."""

from __future__ import annotations

import io
import json
import random
from datetime import date, datetime
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from .synth import Customer, TruthOrder

W, H = A4
DISTRIBUTOR = ("Demo Office Supplies", "orders@demo-office-supplies.example")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_date(iso: str, lang: str, rng: random.Random) -> str:
    d = date.fromisoformat(iso)
    if lang == "fr":
        return d.strftime("%d/%m/%Y")
    return rng.choice([f"{d.day} {MONTHS[d.month - 1]} {d.year}", d.strftime("%d/%m/%Y")])


def fmt_price(x: float, lang: str) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".", ",") if lang == "fr" else f"{x:,.2f}"


def fmt_qty(q: float) -> str:
    return str(int(q)) if q == int(q) else f"{q:g}"


# ---------- email body ----------

def body_lines(o: TruthOrder, c: Customer, rng: random.Random) -> list[str]:
    style = rng.randrange(4)
    out = []
    for ln in o.lines:
        q, t = fmt_qty(ln.quantity), ln.text
        ref = f"{ln.customer_ref} " if ln.customer_ref else ""
        if c.lang == "fr":
            out.append([f"- {q} x {ref}{t}", f"{ref}{t} : {q}", f"{q} {ref}{t}", f"Qte {q} - {ref}{t}"][style])
        else:
            out.append([f"- {q} x {ref}{t}", f"{ref}{t} - qty {q}", f"{q} {ref}{t}", f"{ref}{t} : {q}"][style])
    return out


def email_body(o: TruthOrder, c: Customer, rng: random.Random, attached: bool) -> str:
    first = c.contact.split()[0]
    deliv = fmt_date(o.delivery_date, c.lang, rng) if o.delivery_date else None
    if c.lang == "fr":
        if attached:
            text = f"Bonjour,\n\nVeuillez trouver ci-joint notre bon de commande {o.po_number}."
        else:
            text = ("Bonjour,\n\n" + rng.choice(["Merci de nous livrer :", "Pourriez-vous nous envoyer :", "Commande :"]) +
                    "\n\n" + "\n".join(body_lines(o, c, rng)) + f"\n\nBon de commande n° {o.po_number}")
        if deliv:
            text += f"\nLivraison souhaitee le {deliv}."
        return text + f"\n\nCordialement,\n{c.contact}\n{c.name}\n"
    if attached:
        text = f"Hi,\n\nPlease find attached our purchase order {o.po_number}."
    else:
        text = ("Hi team,\n\n" + rng.choice(["Please could you send the following:", "Can we order:", "Order below please:"]) +
                "\n\n" + "\n".join(body_lines(o, c, rng)) + f"\n\nPO: {o.po_number}")
    if deliv:
        text += f"\nDelivery by {deliv} please."
    return text + f"\n\nThanks,\n{first}\n{c.name}\n"


# ---------- purchase order PDF ----------

def render_po(o: TruthOrder, c: Customer, rng: random.Random) -> bytes:
    fr = c.lang == "fr"
    buf = io.BytesIO()
    cv = Canvas(buf, pagesize=A4)
    accent = HexColor(rng.choice(["#1f4e79", "#2e5e3e", "#5b2c6f", "#333333"]))
    cv.setFillColor(accent)
    cv.setFont("Helvetica-Bold", 18)
    cv.drawString(40, H - 60, c.name)
    cv.setFont("Helvetica", 9)
    cv.setFillColor(HexColor("#444444"))
    cv.drawString(40, H - 75, f"{c.contact}  |  {c.contact.split()[0].lower()}@{c.domain}")
    cv.setFillColor(accent)
    cv.setFont("Helvetica-Bold", 16)
    cv.drawRightString(W - 40, H - 60, "BON DE COMMANDE" if fr else "PURCHASE ORDER")
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica", 10)
    y = H - 110
    rows = [("N° commande" if fr else "PO Number", o.po_number), ("Date", fmt_date(o.order_date, c.lang, rng))]
    if o.delivery_date:
        rows.append(("Livraison souhaitee" if fr else "Delivery date", fmt_date(o.delivery_date, c.lang, rng)))
    for k, v in rows:
        cv.drawString(W - 230, y, f"{k}:")
        cv.drawRightString(W - 40, y, v)
        y -= 15
    cv.drawString(40, H - 110, "Fournisseur :" if fr else "Supplier:")
    cv.setFont("Helvetica-Bold", 10)
    cv.drawString(40, H - 125, DISTRIBUTOR[0])

    y = H - 190
    has_ref = any(ln.customer_ref for ln in o.lines)
    cols = ([("Ref", 40)] if has_ref else []) + [("Designation" if fr else "Description", 100 if has_ref else 40),
                                                  ("Qte" if fr else "Qty", 390)]
    if o.with_prices:
        cols += [("P.U." if fr else "Unit price", 440), ("Montant" if fr else "Amount", 510)]
    cv.setFillColor(accent)
    cv.rect(35, y - 4, W - 70, 18, fill=1, stroke=0)
    cv.setFillColor(HexColor("#ffffff"))
    cv.setFont("Helvetica-Bold", 9)
    for name, x in cols:
        cv.drawString(x, y + 1, name)
    cv.setFillColor(HexColor("#000000"))
    cv.setFont("Helvetica", 9)
    y -= 20
    total = 0.0
    for ln in o.lines:
        x_desc = 100 if has_ref else 40
        if has_ref:
            cv.drawString(40, y, ln.customer_ref or "")
        size = 9.0
        while size > 5.5 and cv.stringWidth(ln.text, "Helvetica", size) > 385 - x_desc:
            size -= 0.25
        cv.setFont("Helvetica", size)
        cv.drawString(x_desc, y, ln.text)
        cv.setFont("Helvetica", 9)
        cv.drawString(390, y, fmt_qty(ln.quantity))
        if o.with_prices and ln.unit_price is not None:
            amount = round(ln.unit_price * ln.quantity, 2)
            total += amount
            cv.drawString(440, y, fmt_price(ln.unit_price, c.lang))
            cv.drawString(510, y, fmt_price(amount, c.lang))
        y -= 17
    if o.with_prices:
        cv.setFont("Helvetica-Bold", 10)
        cv.drawString(440, y - 8, "Total HT" if fr else "Total")
        cv.drawString(510, y - 8, fmt_price(round(total, 2), c.lang))
    cv.setFont("Helvetica", 8)
    cv.setFillColor(HexColor("#666666"))
    cv.drawString(40, 50, "Merci de confirmer la commande par retour de mail." if fr else "Please confirm receipt of this order by email.")
    cv.showPage()
    cv.save()
    return buf.getvalue()


def degrade_to_photo(pdf_bytes: bytes, rng: random.Random) -> bytes:
    """Phone photo of a printed PO: lower resolution, slight rotation, uneven light, blur, noise, JPEG."""
    page = pdfium.PdfDocument(pdf_bytes)[0]
    img = page.render(scale=110 / 72).to_pil().convert("RGB")
    img = img.rotate(rng.uniform(-2.5, 2.5), expand=True, fillcolor=(214, 208, 196), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    h, w, _ = arr.shape
    shade = np.linspace(1.0, rng.uniform(0.78, 0.9), w)[None, :, None] * np.linspace(rng.uniform(0.9, 1.0), 1.0, h)[:, None, None]
    arr = arr * shade + np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 0.9)))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=62)
    return out.getvalue()


# ---------- email ----------

def render_email(o: TruthOrder, c: Customer, rng: random.Random) -> bytes:
    msg = EmailMessage()
    msg["From"] = f"{c.contact} <{o.sender}>"
    msg["To"] = DISTRIBUTOR[1]
    subj = {"fr": ["Commande {po}", "Bon de commande {po}", "Nouvelle commande"],
            "en": ["Order {po}", "PO {po}", "New order"]}[c.lang]
    msg["Subject"] = rng.choice(subj).format(po=o.po_number)
    d = date.fromisoformat(o.order_date)
    msg["Date"] = format_datetime(datetime(d.year, d.month, d.day, rng.randint(7, 17), rng.randint(0, 59)))
    attached = o.format in ("pdf", "photo")
    msg.set_content(email_body(o, c, rng, attached))
    if attached:
        pdf = render_po(o, c, rng)
        name = f"{'BC' if c.lang == 'fr' else 'PO'}_{o.po_number.replace('/', '-')}"
        if o.format == "pdf":
            msg.add_attachment(pdf, maintype="application", subtype="pdf", filename=f"{name}.pdf")
        else:
            msg.add_attachment(degrade_to_photo(pdf, rng), maintype="image", subtype="jpeg", filename=f"IMG_{rng.randint(1000, 9999)}.jpg")
    return bytes(msg)


def write_batch(orders: list[TruthOrder], customers: list[Customer], out_dir: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    by_id = {c.id: c for c in customers}
    (out_dir / "truth").mkdir(parents=True, exist_ok=True)
    for o in orders:
        (out_dir / f"{o.order_id}.eml").write_bytes(render_email(o, by_id[o.customer_id], rng))
        (out_dir / "truth" / f"{o.order_id}.json").write_text(json.dumps(o.model_dump(), indent=1, ensure_ascii=False))
