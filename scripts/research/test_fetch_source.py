"""fetch_source.paragraphs / propose on converted PDF text (acceptance 2026-10-30 of design 2026-10-29, N18).

pdftotext -layout puts each page footer before a form feed and the next page's running header after it. A sentence
cut by that page break used to end its paragraph, and a first half without a template anchor was never proposed.
No network, no pdftotext: the converted text is written here.
  PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_fetch_source.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
import fetch_source  # noqa: E402

FOOT = "© 2026 Example Memory, Inc.                                                   September 30, 2026\n"
TOP = "\f                                                                   Example Memory, Inc.\n"
TOP += "                                                  Fiscal Q4 2026 Earnings Call Prepared Remarks\n\n\n\n"
# The shape of the Micron FQ4 FY26 prepared remarks, lines 186-207 and 455-465.
TEXT = (
    "Physical AI\n\n"
    "Autonomous vehicles are the first major deployment of physical AI. Memory content in Level 4 and higher\n"
    "autonomous vehicles typically exceeds 200GB, each more than an order of magnitude greater than in today's Level\n"
    "\n\n\n" + FOOT + TOP + "2+ and Level 3 semi-autonomous vehicles. Physical AI can become a significant driver of\n"
    "memory and storage demand by the end of this decade.\n\n"
    "Market outlook\n\n"
    "We expect memory and storage supply-demand conditions to be much tighter in calendar 2027 and 2028 than they\n"
    "were in 2026.\n"
    "\n\n\n" + FOOT + TOP + "Revenue\n\n"
    "Revenue was a record $37.8 billion for the quarter, up 46% from the prior quarter on higher pricing and\n"
    "favorable mix.\n\n\n" + FOOT + "\f"
)


def converted(tmp_path):
    path = tmp_path / "remarks.pdf.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


def test_a_sentence_cut_by_a_page_footer_is_one_paragraph(tmp_path):
    path = converted(tmp_path)
    paras = {text: (s, e) for s, e, text in fetch_source.paragraphs(path)}
    (joined,) = [t for t in paras if "200GB" in t]
    assert "today's Level © 2026 Example Memory" in joined and joined.endswith("by the end of this decade.")
    assert paras[joined] == (3, 15)
    # the first half has no earnings anchor; joined, the paragraph is proposed and still verbatim in the file
    (proposal,) = [p for p in fetch_source.propose(path, "earnings") if "200GB" in p["text"]]
    assert proposal["verified"] and proposal["locator"] == "lines 3-15" and "demand|需求" in proposal["anchors"]


def test_a_paragraph_that_ends_its_sentence_before_the_footer_is_not_joined(tmp_path):
    paras = fetch_source.paragraphs(converted(tmp_path))
    outlook = next(p for p in paras if p[2].startswith("We expect"))
    assert outlook[:2] == (19, 20) and outlook[2].endswith("were in 2026.")
    # as before: the footer is dropped and the running header is a paragraph of its own
    assert not any(t.startswith("©") for _, _, t in paras)
    assert (25, 26, "Example Memory, Inc. Fiscal Q4 2026 Earnings Call Prepared Remarks") in paras
