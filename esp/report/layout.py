"""Shared page furniture for the PDFs this repo produces.

Two documents are built from here -- the run report and the repository dossier
-- and they have to look like one piece of work rather than two. Keeping the
palette, the tables and the terminal blocks in one place is what makes that
true by construction instead of by care.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#101418")
MUTED = colors.HexColor("#5b6672")
RULE = colors.HexColor("#d7dde3")
ACCENT = colors.HexColor("#1a4fa0")
SOFT = colors.HexColor("#eaf0fa")
GOOD = colors.HexColor("#1d7a46")
GOOD_BG = colors.HexColor("#eaf6ef")
BAD = colors.HexColor("#b3261e")
BAD_BG = colors.HexColor("#fdeceb")
WARN_BG = colors.HexColor("#fff6e5")
AMBER = colors.HexColor("#b8860b")
TERM_BG = colors.HexColor("#12161b")

PW, PH = A4
MARGIN = 18 * mm
CW = PW - 2 * MARGIN


def _s(name, **kw):
    base = dict(fontName="Helvetica", fontSize=10, leading=14.6,
                textColor=INK, spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


BODY = _s("body", alignment=TA_JUSTIFY)
LEAD = _s("lead", fontSize=11.4, leading=17, textColor=colors.HexColor("#26313c"))
H1 = _s("h1", fontName="Helvetica-Bold", fontSize=19, leading=23,
        textColor=ACCENT, spaceAfter=2)
H2 = _s("h2", fontName="Helvetica-Bold", fontSize=13, leading=16.5,
        spaceBefore=14, spaceAfter=4)
CAP = _s("cap", fontSize=8.4, leading=11.6, textColor=MUTED, alignment=1,
         spaceBefore=3, spaceAfter=12)
MONO = ParagraphStyle("mono", fontName="Courier", fontSize=7.6, leading=10,
                      textColor=colors.HexColor("#d6e2ee"))


def _hx(c):
    return "#" + c.hexval()[2:]


class Layout:
    """Story-building helpers. A document subclasses this and writes `build`."""

    def __init__(self, results_dir: str | Path = "results"):
        self.dir = Path(results_dir)
        self.story: list = []

    # --------------------------------------------------------------- helpers

    def h1(self, text, sub=None):
        self.story.append(Paragraph(text, H1))
        self.story.append(self._rule(ACCENT, 1.3))
        self.story.append(Paragraph(sub, _s("sub", fontSize=10, textColor=MUTED,
                                            spaceAfter=13)) if sub
                          else Spacer(1, 9))

    def h2(self, text):
        self.story.append(Paragraph(text, H2))

    def p(self, text, style=BODY):
        self.story.append(Paragraph(text, style))

    def _rule(self, color=RULE, w=0.6):
        t = Table([[""]], colWidths=[CW], rowHeights=[w])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color),
                               ("TOPPADDING", (0, 0), (-1, -1), 0),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return t

    def callout(self, title, text, bg=SOFT, bar=ACCENT):
        inner = []
        if title:
            inner.append(Paragraph(f'<font color="{_hx(bar)}"><b>{title}</b></font>',
                                   _s("ct", fontSize=9.4, leading=12.6, spaceAfter=3)))
        inner.append(Paragraph(text, _s("cb", fontSize=9.4, leading=13.6, spaceAfter=0)))
        t = Table([[inner]], colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ]))
        self.story += [t, Spacer(1, 10)]

    def terminal(self, text):
        t = Table([[Preformatted(text.rstrip(), MONO)]], colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), TERM_BG),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        self.story += [t, Spacer(1, 10)]

    def table(self, header, rows, widths=None, highlight=None):
        style = _s("td", fontSize=8.6, leading=11.6, spaceAfter=0)
        head = _s("th", fontSize=8.6, leading=11.6, spaceAfter=0,
                  fontName="Helvetica-Bold", textColor=colors.white)
        data = [[Paragraph(str(c), head) for c in header]]
        data += [[Paragraph(str(c), style) for c in row] for row in rows]
        widths = widths or [CW / len(header)] * len(header)
        t = Table(data, colWidths=widths, repeatRows=1)
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
        for index in (highlight or []):
            commands.append(("BACKGROUND", (0, index + 1), (-1, index + 1), GOOD_BG))
        t.setStyle(TableStyle(commands))
        self.story += [t, Spacer(1, 10)]

    def image(self, path, caption, width=None) -> bool:
        """Place an image from an explicit path. Returns whether it landed.

        Separate from `figure`, which resolves inside results/figures and
        returns silently when a file is missing. That is fine for an optional
        plot and wrong for a screenshot offered as evidence -- a caller can act
        on the False and say the evidence is absent.
        """
        path = Path(path)
        if not path.exists():
            return False
        from PIL import Image as PILImage
        iw, ih = PILImage.open(path).size
        w = width or CW
        img = Image(str(path), width=w, height=w * ih / iw)
        img.hAlign = "CENTER"
        self.story += [img, Paragraph(caption, CAP)]
        return True

    def figure(self, name, caption, width=None):
        path = self.dir / "figures" / name
        if not path.exists():
            return
        from PIL import Image as PILImage
        iw, ih = PILImage.open(path).size
        w = width or CW
        img = Image(str(path), width=w, height=w * ih / iw)
        img.hAlign = "CENTER"
        self.story += [img, Paragraph(caption, CAP)]

    def render(self, out: Path, title: str) -> Path:
        doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN,
                              rightMargin=MARGIN, topMargin=MARGIN,
                              bottomMargin=MARGIN, title=title,
                              author="Sivakumar Raj")
        frame = Frame(MARGIN, MARGIN, CW, PH - 2 * MARGIN, id="f")
        doc.addPageTemplates([PageTemplate(id="n", frames=[frame],
                                           onPage=_page_furniture)])
        doc.build(self.story)
        return out


def _page_furniture(canv, doc):
    canv.saveState()
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.5)
    canv.line(MARGIN, PH - MARGIN + 6 * mm, PW - MARGIN, PH - MARGIN + 6 * mm)
    canv.setFont("Helvetica", 7.4)
    canv.setFillColor(MUTED)
    canv.drawString(MARGIN, PH - MARGIN + 8.4 * mm,
                    "NEURO-SAN-ESP  \u2014  EVOLVING AGENT NETWORKS")
    canv.drawRightString(PW - MARGIN, PH - MARGIN + 8.4 * mm,
                         "github.com/Sivakumarraj/neuro-san-esp")
    canv.line(MARGIN, MARGIN - 6 * mm, PW - MARGIN, MARGIN - 6 * mm)
    canv.drawCentredString(PW / 2, MARGIN - 10 * mm, str(canv.getPageNumber()))
    canv.restoreState()
