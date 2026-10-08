"""Review screen. Run: uv run streamlit run app.py"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
import streamlit as st

from order_intake_ai.export import ERP_COLUMNS, erp_rows
from order_intake_ai.extract import read_eml
from order_intake_ai.match import Memory

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "out"
INBOX = DATA / "samples"

st.set_page_config(page_title="Order Intake AI", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.issue {background:#fff4f2;border-left:4px solid #d93025;padding:6px 10px;margin:4px 0;border-radius:4px}
.ok {background:#eef8f1;border-left:4px solid #17703a;padding:6px 10px;margin:4px 0;border-radius:4px}
</style>""", unsafe_allow_html=True)

st.title("Order Intake AI")
st.caption("Customer order emails, PDFs and phone photos to ERP-ready orders. Every line is matched to a catalog "
           "SKU; doubtful lines come here with the top 3 candidates. Runs locally: no order leaves the machine.")


@st.cache_resource(show_spinner="Loading catalog and embedding model...")
def get_matcher():
    from order_intake_ai.catalog import load_catalog
    from order_intake_ai.match import Matcher

    cfg = json.loads((DATA / "matcher.json").read_text())
    return Matcher(load_catalog(DATA / "catalog.json"), tau=cfg["tau"], margin=cfg["margin"])


with st.sidebar:
    st.subheader("Inbox")
    st.write(f"{len(list(INBOX.glob('*.eml')))} order emails in `data/samples`")
    model = st.text_input("Local model", "qwen2.5vl")
    use_memory = st.toggle("Use customer memory", value=True,
                           help="Wordings and article codes staff already confirmed for each customer.")
    go = st.button("Process inbox", type="primary", width="stretch")
    st.divider()
    st.markdown("**Matching cascade**\n1. our SKU or manufacturer code\n2. customer memory\n"
                "3. text + multilingual embeddings + attributes\n\n**Checks**\n- product not in catalog\n"
                "- discontinued (replacement proposed)\n- price vs customer price list\n- quantity vs order history\n"
                "- duplicate PO\n- unknown sender")

if go:
    from order_intake_ai.pipeline import run

    with st.status("Processing locally...", expanded=True) as s:
        rep = run(INBOX, OUT, DATA, INBOX / "truth", model=model, use_memory=use_memory, log=s.write, matcher=get_matcher())
        s.update(label="Done", state="complete")
    st.session_state["report"] = rep
elif "report" not in st.session_state and (OUT / "report.json").exists():
    st.session_state["report"] = json.loads((OUT / "report.json").read_text())

report = st.session_state.get("report")
if not report:
    st.info("Press **Process inbox**.")
    st.stop()

orders = report["orders"]
summary = report.get("summary")
lines_all = [ln for o in orders for ln in o["lines"]]
auto = sum(ln["match"]["status"] == "AUTO" for ln in lines_all)

k = st.columns(5)
k[0].metric("Orders", len(orders))
k[1].metric("Ready for ERP", sum(o["status"] == "APPROVED" for o in orders))
k[2].metric("Lines matched automatically", f"{auto / len(lines_all):.0%}" if lines_all else "-")
if summary:
    k[3].metric("Wrong automatic matches", summary["wrong_auto_matches"])
    k[4].metric("Planted problems caught", f"{summary['anomalies_caught']}/{summary['anomalies_total']}")

table = pd.DataFrame([{
    "File": o["file"], "Status": o["status"], "Customer": o["customer_name"] or o["sender"], "PO": o["po_number"],
    "Lines": len(o["lines"]), "Auto-matched": sum(ln["match"]["status"] == "AUTO" for ln in o["lines"]),
    "Source": o["route"], "Seconds": o["seconds"],
    "Issues": len(o["issues"]) + sum(len(ln["issues"]) for ln in o["lines"]),
} for o in orders])
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Status": st.column_config.TextColumn(width="small")})

st.divider()
review_first = sorted(orders, key=lambda o: (o["status"] != "REVIEW", o["file"]))
files = [o["file"] for o in review_first]
wanted = st.query_params.get("order")  # deep link: ?order=ord_006.eml
pick = st.selectbox("Open order", files, index=files.index(wanted) if wanted in files else 0,
                    format_func=lambda f: next(f"{o['status']:8}  {f}  {o['customer_name'] or o['sender']}  PO {o['po_number']}"
                                               for o in orders if o["file"] == f))
o = next(x for x in orders if x["file"] == pick)
left, right = st.columns([5, 4])

with right:
    mail = read_eml(INBOX / o["file"])
    st.markdown(f"**From** {mail.sender}  \n**Subject** {mail.subject}")
    if mail.attachment_type == "image":
        st.image(mail.attachment, width="stretch")
    elif mail.attachment_type == "pdf":
        page = pdfium.PdfDocument(mail.attachment)[0]
        buf = io.BytesIO()
        page.render(scale=1.4).to_pil().save(buf, format="PNG")
        st.image(buf.getvalue(), width="stretch")
    with st.expander("Email body", expanded=mail.attachment is None):
        st.text(mail.body)

with left:
    st.markdown(f"### {o['customer_name'] or o['sender']}\nPO **{o['po_number']}** · ordered {o['order_date']} · "
                f"delivery {o['delivery_date'] or '-'} · read via `{o['route']}` in {o['seconds']:.0f}s")
    for x in o["issues"]:
        st.markdown(f"<div class='issue'>{x['message']}</div>", unsafe_allow_html=True)
    memory = Memory(DATA / "memory.json")
    for ln in o["lines"]:
        m = ln["match"]
        head = f"**{ln['line_no']}.** `{ln['description']}` × **{ln['quantity']:g}**" if ln["quantity"] else f"**{ln['line_no']}.** `{ln['description']}`"
        st.markdown(head)
        if m["status"] == "AUTO" and not ln["issues"]:
            st.markdown(f"<div class='ok'>→ {m['sku']} {m['name']} &nbsp;·&nbsp; {m['method']}, "
                        f"score {m['score']:.2f}</div>", unsafe_allow_html=True)
            continue
        for x in ln["issues"]:
            st.markdown(f"<div class='issue'>{x['message']}</div>", unsafe_allow_html=True)
        opts = [f"{c['sku']}  {c['name']}  ({c['score']:.2f})" for c in m["candidates"]]
        default = next((i for i, c in enumerate(m["candidates"]) if c["sku"] == m["sku"]), 0)
        c1, c2 = st.columns([4, 1])
        choice = c1.selectbox("Product", opts + ["Not in catalog / skip"], index=default, key=f"{pick}-{ln['line_no']}",
                              label_visibility="collapsed")
        if c2.button("Confirm", key=f"ok-{pick}-{ln['line_no']}") and o["customer_id"] and not choice.startswith("Not in"):
            sku = choice.split()[0]
            memory.learn(o["customer_id"], ln["description"], ln["customer_ref"], sku)
            memory.save()
            c2.success("Learned")

st.divider()
rows = erp_rows(report)
c1, c2 = st.columns(2)
c1.download_button("Download ERP import (CSV)", pd.DataFrame(rows, columns=ERP_COLUMNS).to_csv(index=False),
                   "erp_import.csv", "text/csv", width="stretch")
if (OUT / "orders.xlsx").exists():
    c2.download_button("Download workbook (Excel)", (OUT / "orders.xlsx").read_bytes(), "orders.xlsx",
                       width="stretch")
