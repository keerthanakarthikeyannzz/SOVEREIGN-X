"""
SOVEREIGN — Industrial AI Analysis Report Generator
Generates a professional 14-section DOCX inspection report.
All processing is local. python-docx only. No external dependencies.
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path
from datetime import datetime
from typing import Optional
from backend.config import get_settings


# ── Helpers ───────────────────────────────────────────────────────────────
def _shade_cell(cell, hex_color: str):
    """Set table cell background shading."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def _first_run(paragraph):
    if not paragraph.runs:
        return paragraph.add_run("")
    return paragraph.runs[0]


def _set_cell(cell, text: str, bold: bool = False, size: int = 9,
              fg: Optional[RGBColor] = None, bg: Optional[str] = None,
              italic: bool = False, align: str = "left"):
    cell.text = str(text) if text is not None else ""
    para = cell.paragraphs[0]
    if align == "center":
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = _first_run(para)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if fg:
        run.font.color.rgb = fg
    if bg:
        _shade_cell(cell, bg)


def _heading(doc: Document, text: str, level: int = 2, color: RGBColor = None):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = _first_run(h)
    if level == 1:
        r.font.size = Pt(16)
    elif level == 2:
        r.font.size = Pt(11)
        r.font.bold = True
    else:
        r.font.size = Pt(10)
        r.font.bold = True
    if color:
        r.font.color.rgb = color
    return h


def _para(doc: Document, text: str, size: int = 10, color: RGBColor = None,
          bold: bool = False, italic: bool = False, indent: float = 0):
    p = doc.add_paragraph(text)
    r = _first_run(p)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    if color:
        r.font.color.rgb = color
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    return p


def _add_rule(doc: Document):
    p = doc.add_paragraph("─" * 90)
    r = _first_run(p)
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(0x60, 0x70, 0x50)


def _section_header(doc: Document, number: str, title: str):
    """Add numbered section heading with olive accent."""
    p = doc.add_paragraph()
    r1 = p.add_run(f"{number}  ")
    r1.font.size = Pt(11)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0x6b, 0x7a, 0x47)  # olive
    r2 = p.add_run(title.upper())
    r2.font.size = Pt(11)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(0x22, 0x2e, 0x1a)  # dark forest
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(3)
    return p


def _kv_table(doc: Document, rows_data: list[tuple], col_widths=(2.2, 4.0)):
    """Create a two-column key-value table."""
    table = doc.add_table(rows=len(rows_data), cols=2)
    table.style = "Table Grid"
    for i, (key, val) in enumerate(rows_data):
        row = table.rows[i]
        _set_cell(row.cells[0], key, bold=True, size=9,
                  fg=RGBColor(0xFF, 0xFF, 0xFF), bg="3a4e2a")
        _set_cell(row.cells[1], str(val) if val else "—", size=9,
                  fg=RGBColor(0x1a, 0x1a, 0x1a))
    return table


def _risk_bar(doc: Document, risk_level: str, confidence: float):
    """ASCII-style risk indicator that works in .docx."""
    level_map = {
        "CRITICAL": ("HIGH SEVERITY — IMMEDIATE ACTION REQUIRED", "C0392B", "■■■■■■■■■■ 100%"),
        "WARNING":  ("MEDIUM SEVERITY — PLANNED ACTION RECOMMENDED", "D4943A", "■■■■■■░░░░  60%"),
        "NORMAL":   ("LOW SEVERITY — ROUTINE MONITORING", "4A7C59",  "■■■░░░░░░░  30%"),
    }
    label, color_hex, bar = level_map.get(risk_level.upper(), ("UNKNOWN", "888888", "░░░░░░░░░░   0%"))
    r, g, b = int(color_hex[:2], 16), int(color_hex[2:4], 16), int(color_hex[4:], 16)
    color = RGBColor(r, g, b)

    table = doc.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    row = table.rows[0]
    _set_cell(row.cells[0], risk_level.upper(), bold=True, size=11,
              fg=color, bg=color_hex + "20"[::-1] if len(color_hex) == 6 else "F8F8F8")
    _set_cell(row.cells[1], label, bold=False, size=9)
    _set_cell(row.cells[2], f"Conf: {confidence:.0%}  {bar}", bold=False, size=9,
              fg=color, align="center")
    return table


# ═══════════════════════════════════════════════════════════════════════════
class DocumentGenerator:
    """Generates a 14-section professional industrial inspection report."""

    def __init__(self):
        self.settings = get_settings()

    def generate_approval_note(
        self,
        mission_id: str,
        mission_data: dict,
        evidence: list,
        analysis: str,
        task_dna: dict = None,
        approval_status: str = "PENDING",
        approver_notes: str = "",
    ) -> str:

        generated_path = self.settings.generated_path
        output_path = generated_path / f"{mission_id}_approval_note.docx"

        doc = Document()

        # ── Page Setup ────────────────────────────────────────────────
        section = doc.sections[0]
        section.page_width  = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin = section.right_margin   = Inches(0.9)
        section.top_margin  = section.bottom_margin  = Inches(0.9)

        # ── Color constants ───────────────────────────────────────────
        WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
        CHARCOAL  = RGBColor(0x1a, 0x22, 0x14)
        OLIVE     = RGBColor(0x6b, 0x7a, 0x47)
        FOREST    = RGBColor(0x3a, 0x4e, 0x2a)
        SAGE      = RGBColor(0x8a, 0x9a, 0x5b)
        RED       = RGBColor(0xC0, 0x39, 0x2B)
        AMBER     = RGBColor(0xD4, 0x94, 0x3A)
        GREEN_OK  = RGBColor(0x3d, 0x7a, 0x52)

        risk_level      = str(mission_data.get("risk_level") or "unknown").upper()
        classification  = str(mission_data.get("classification") or "UNKNOWN")
        confidence      = float(mission_data.get("confidence_score") or 0)
        equipment_id    = str(mission_data.get("equipment_id") or "N/A")
        equipment_type  = str(mission_data.get("equipment_type") or "N/A")
        description     = str(mission_data.get("description") or "No description provided.")
        recommendation  = str(mission_data.get("recommendation") or "No recommendation generated.")

        now_str   = datetime.utcnow().strftime("%d %b %Y  %H:%M UTC")
        risk_color = RED if classification == "CRITICAL" else AMBER if classification == "WARNING" else GREEN_OK

        # ══════════════════════════════════════════════════════════════
        # COVER PAGE
        # ══════════════════════════════════════════════════════════════

        # Wordmark
        p_mark = doc.add_paragraph()
        p_mark.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_mark.add_run("SOVEREIGN")
        r.font.size = Pt(28)
        r.font.bold = True
        r.font.color.rgb = FOREST

        p_sub = doc.add_paragraph()
        p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_sub.add_run("INDUSTRIAL AI ANALYSIS REPORT")
        r.font.size = Pt(11)
        r.font.color.rgb = OLIVE
        r.font.bold = True

        p_org = doc.add_paragraph()
        p_org.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_org.add_run("Mangalore Refinery & Petrochemicals Limited  |  Project 26117  |  On-Premise Agentic AI Workbench")
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x80, 0x90, 0x70)

        doc.add_paragraph()  # spacer

        # Cover summary table
        cover_table = doc.add_table(rows=5, cols=4)
        cover_table.style = "Table Grid"
        cover_table.alignment = WD_TABLE_ALIGNMENT.CENTER

        cover_data = [
            ("Mission ID",       mission_id,    "Report Date",      now_str),
            ("Equipment ID",     equipment_id,  "Equipment Type",   equipment_type),
            ("Classification",   classification,"Risk Level",       risk_level),
            ("Approval Status",  approval_status, "AI Confidence",  f"{confidence * 100:.0f}%"),
            ("Sovereignty",      "LOCAL ONLY",  "Network",         "AIR-GAPPED — No data left premises"),
        ]

        status_colors = {"CRITICAL": "FADBD8", "WARNING": "FEF9E7", "NORMAL": "D5F5E3"}
        cls_bg = status_colors.get(classification, "F8F9FA")

        for ri, row_data in enumerate(cover_data):
            row = cover_table.rows[ri]
            for ci, text in enumerate(row_data):
                is_label = (ci % 2 == 0)
                bg = "3a4e2a" if is_label else None
                if ri == 2 and ci == 1:
                    bg = cls_bg
                _set_cell(row.cells[ci], text, bold=is_label,
                          fg=WHITE if is_label else CHARCOAL, bg=bg, size=9)

        doc.add_paragraph()
        _add_rule(doc)
        doc.add_page_break()

        # ══════════════════════════════════════════════════════════════
        # SECTION 1 — EXECUTIVE SUMMARY
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "01", "Executive Summary")
        _kv_table(doc, [
            ("Mission ID",       mission_id),
            ("Equipment",        f"{equipment_id} — {equipment_type}"),
            ("AI Classification",f"{classification} | Risk: {risk_level}"),
            ("Confidence Score", f"{confidence * 100:.0f}%"),
            ("Evidence Sources", f"{len(evidence)} documents retrieved from knowledge base"),
            ("Approval Status",  approval_status),
            ("Generated",        now_str),
        ])

        _para(doc, "", 6)  # spacer
        p_key = doc.add_paragraph()
        r = p_key.add_run("KEY FINDING:  ")
        r.font.bold = True; r.font.size = Pt(9); r.font.color.rgb = risk_color
        r2 = p_key.add_run(recommendation.split("\n")[0][:300])
        r2.font.size = Pt(9); r2.font.color.rgb = CHARCOAL

        # ══════════════════════════════════════════════════════════════
        # SECTION 2 — EQUIPMENT / ASSET INFORMATION
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "02", "Equipment / Asset Information")
        _kv_table(doc, [
            ("Equipment ID",       equipment_id),
            ("Equipment Type",     equipment_type),
            ("Operating Location", "MRPL Refinery — Mangalore"),
            ("Task Domain",        task_dna.get("domain") if task_dna else "Industrial Inspection"),
        ])

        # ══════════════════════════════════════════════════════════════
        # SECTION 3 — INSPECTION INPUT
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "03", "Inspection Input")
        _para(doc, description, size=9)

        if evidence:
            _para(doc, f"\nInput Documents ({len(evidence)} files retrieved by RAG):", size=9, bold=True)
            for ev in evidence[:5]:
                _para(doc, f"  •  {ev.get('filename', '—')}  [{ev.get('source_type', '')}]  —  Relevance: {(ev.get('score', 0) or 0)*100:.0f}%",
                      size=8)

        # ══════════════════════════════════════════════════════════════
        # SECTION 4 — DETECTED PROBLEM
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "04", "Detected Problem")

        box_table = doc.add_table(rows=1, cols=1)
        box_table.style = "Table Grid"
        cell = box_table.rows[0].cells[0]
        _shade_cell(cell, cls_bg if classification in status_colors else "F8F9FA")
        cell.text = ""
        r_label = cell.paragraphs[0].add_run("PROBLEM STATEMENT\n")
        r_label.font.size = Pt(7); r_label.font.bold = True; r_label.font.color.rgb = risk_color
        r_body = cell.paragraphs[0].add_run(recommendation.split("\n")[0][:400])
        r_body.font.size = Pt(10); r_body.font.bold = True; r_body.font.color.rgb = CHARCOAL

        # ══════════════════════════════════════════════════════════════
        # SECTION 5 — SEVERITY / RISK ASSESSMENT
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "05", "Severity / Risk Assessment")
        _risk_bar(doc, risk_level, confidence)
        _para(doc, "", 4)
        _kv_table(doc, [
            ("Classification",  classification),
            ("Risk Level",      risk_level),
            ("AI Confidence",   f"{confidence * 100:.0f}%  (model certainty in this finding)"),
        ])

        # ══════════════════════════════════════════════════════════════
        # SECTION 6 — EVIDENCE & SUPPORTING DATA
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "06", "Evidence & Supporting Data")

        if evidence:
            ev_table = doc.add_table(rows=1, cols=5)
            ev_table.style = "Table Grid"
            headers = ["Rank", "Filename", "Source Type", "Doc Type", "Relevance"]
            for ci, h in enumerate(headers):
                _set_cell(ev_table.rows[0].cells[ci], h, bold=True,
                          fg=WHITE, bg="3a4e2a", size=8)

            src_colors = {
                "MRPL_PUBLIC":    "D5F5E3",
                "SYNTHETIC_DEMO": "EAF2F8",
                "USER_UPLOAD":    "FEF9E7",
            }
            for ev in evidence[:15]:
                row = ev_table.add_row()
                score = ev.get("score", 0) or 0
                _set_cell(row.cells[0], str(ev.get("rank", "")), size=8)
                _set_cell(row.cells[1], str(ev.get("filename") or ""), size=8)
                src = str(ev.get("source_type") or "")
                _set_cell(row.cells[2], src, size=8, bg=src_colors.get(src))
                _set_cell(row.cells[3], str(ev.get("document_type") or ""), size=8)
                _set_cell(row.cells[4], f"{score * 100:.0f}%", size=8,
                          fg=GREEN_OK if score > 0.7 else AMBER if score > 0.4 else RED)
        else:
            _para(doc, "No evidence retrieved from the knowledge base.", size=9, italic=True)

        # ══════════════════════════════════════════════════════════════
        # SECTION 7 — TECHNICAL FINDINGS (full AI analysis)
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "07", "Technical Findings — AI Analysis Output")
        if analysis and analysis.strip():
            for line in analysis.strip().split("\n")[:80]:
                stripped = line.strip()
                if not stripped:
                    doc.add_paragraph()
                    continue
                is_header = stripped.endswith(":") and len(stripped) < 60
                _para(doc, stripped, size=9, bold=is_header,
                      color=FOREST if is_header else None)
        else:
            _para(doc, "Full analysis text not available. Run mission and re-generate.", size=9, italic=True)

        # ══════════════════════════════════════════════════════════════
        # SECTION 8 — DAMAGE / POTENTIAL IMPACT
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "08", "Damage / Potential Impact")
        impact_map = {
            "CRITICAL": (
                "Unaddressed CRITICAL findings carry significant risk of equipment failure, production loss, "
                "HSE incidents, or escalating structural damage. Immediate intervention is advised.\n\n"
                "Potential consequences if not actioned:\n"
                "  • Unplanned shutdown and production impact\n"
                "  • Safety hazard to personnel (containment failure, fire, explosion risk)\n"
                "  • Regulatory compliance violation\n"
                "  • Progressive damage leading to total asset loss"
            ),
            "WARNING": (
                "WARNING-level findings indicate a developing condition that, if left unaddressed, "
                "may escalate to critical failure. Planned maintenance is recommended within the next inspection cycle.\n\n"
                "Potential consequences if not actioned:\n"
                "  • Accelerated degradation and reduced asset life\n"
                "  • Process inefficiency and off-spec product risk\n"
                "  • Unplanned failure during next high-load period"
            ),
            "NORMAL": (
                "NORMAL findings indicate equipment operating within acceptable parameters. "
                "Continue standard monitoring schedule and routine maintenance plan."
            ),
        }
        _para(doc, impact_map.get(classification, "Impact assessment not available for this classification level."), size=9)

        # ══════════════════════════════════════════════════════════════
        # SECTION 9 — ROOT CAUSE / POSSIBLE CAUSE
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "09", "Root Cause / Possible Cause")
        task_type = (task_dna.get("task_type") if task_dna else None) or "inspection"
        _para(doc, f"AI Analysis Domain: {task_type.replace('_', ' ').title()}", size=8, bold=True)
        _para(doc,
              "The root cause was identified through cross-referencing inspection observations "
              "with historical SOP thresholds, OEM specifications, and maintenance records "
              "retrieved from the local knowledge base. Refer to the full AI Analysis Output "
              "(Section 07) for the detailed causal chain identified by the reasoning model.",
              size=9)
        if task_dna:
            _para(doc, "\nAnalysis Capability Stack Used:", size=8, bold=True)
            for label, val in [
                ("OCR Applied",       task_dna.get("requires_ocr")),
                ("Vision Analysis",   task_dna.get("requires_vision")),
                ("RAG Retrieval",     task_dna.get("requires_rag")),
                ("Historical Data",   task_dna.get("requires_historical_analysis")),
                ("Reasoning Model",   task_dna.get("selected_reasoning_model") or "N/A"),
            ]:
                _para(doc, f"  {'✓' if val else '○'}  {label}: {val if isinstance(val, str) else ('Yes' if val else 'No')}", size=8)

        # ══════════════════════════════════════════════════════════════
        # SECTION 10 — RECOMMENDED CORRECTIVE ACTION
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "10", "Recommended Corrective Action")

        rec_table = doc.add_table(rows=1, cols=1)
        rec_table.style = "Table Grid"
        rec_cell = rec_table.rows[0].cells[0]
        _shade_cell(rec_cell, "EAF2E8")
        rec_cell.text = ""
        r_h = rec_cell.paragraphs[0].add_run("SOVEREIGN RECOMMENDATION\n")
        r_h.font.size = Pt(7); r_h.font.bold = True; r_h.font.color.rgb = FOREST
        r_t = rec_cell.paragraphs[0].add_run(recommendation)
        r_t.font.size = Pt(9); r_t.font.color.rgb = CHARCOAL

        # ══════════════════════════════════════════════════════════════
        # SECTION 11 — PRIORITY / URGENCY
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "11", "Priority / Urgency")
        urgency_map = {
            "CRITICAL": "P1 — IMMEDIATE  (within 24 hours or next safe opportunity)",
            "WARNING":  "P2 — PLANNED    (within current or next maintenance window)",
            "NORMAL":   "P3 — ROUTINE    (standard monitoring cycle)",
        }
        _kv_table(doc, [
            ("Priority Level",  urgency_map.get(classification, "P4 — MONITOR")),
            ("Classification",  classification),
            ("Recommended Timeline", urgency_map.get(classification, "Refer to maintenance schedule")),
        ])

        # ══════════════════════════════════════════════════════════════
        # SECTION 12 — VERIFICATION / CONFIDENCE
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "12", "Verification / Confidence")
        _kv_table(doc, [
            ("AI Confidence Score",    f"{confidence * 100:.0f}%"),
            ("Evidence Sources Used",  f"{len(evidence)} documents"),
            ("RAG Collections",        ", ".join(task_dna.get("rag_collections") or ["mrpl_sop", "industrial_docs"]) if task_dna else "mrpl_sop, industrial_docs"),
            ("Reasoning Model",        task_dna.get("selected_reasoning_model") if task_dna else "N/A"),
            ("Verification Note",      "AI findings require qualified human review before operational action."),
        ])

        # ══════════════════════════════════════════════════════════════
        # SECTION 13 — SAFETY CONSIDERATIONS
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "13", "Safety Considerations")
        _para(doc, "Before any physical intervention on this equipment:", size=9, bold=True, color=RED)
        for bullet in [
            "Obtain appropriate Work Permit from the Permit-to-Work authority.",
            "Apply Lockout/Tagout (LOTO) procedure per General Maintenance Safety SOP.",
            "Verify zero-energy state before maintenance begins.",
            "Wear appropriate PPE (hard hat, safety glasses, gloves, FRC as applicable).",
            "Refer to the relevant equipment-specific Safety SOP for detailed controls.",
            "AI recommendations do NOT replace qualified human engineering review.",
        ]:
            _para(doc, f"  •  {bullet}", size=9, indent=0.2)

        # ══════════════════════════════════════════════════════════════
        # SECTION 14 — HUMAN APPROVAL
        # ══════════════════════════════════════════════════════════════
        _section_header(doc, "14", "Human Approval")
        approved_at_str = datetime.utcnow().strftime("%d %b %Y %H:%M UTC") if approval_status != "PENDING" else "—"
        _kv_table(doc, [
            ("Approval Status",    approval_status),
            ("Approver Notes",     approver_notes or "Pending review"),
            ("Approved At",        approved_at_str),
            ("Approver",           "Authorised Engineer — MRPL"),
            ("Approval Required",  "Yes — AI recommendations require human sign-off before implementation"),
        ])

        doc.add_paragraph()

        # Signature block
        sig_table = doc.add_table(rows=3, cols=3)
        sig_table.style = "Table Grid"
        headers_sig = ["Reviewed By", "Approved By", "Date"]
        for i, h in enumerate(headers_sig):
            _set_cell(sig_table.rows[0].cells[i], h, bold=True,
                      fg=WHITE, bg="3a4e2a", size=8)
            _set_cell(sig_table.rows[1].cells[i], " ", size=16)  # signature space
            _set_cell(sig_table.rows[2].cells[i], "Name / Designation", size=7,
                      fg=RGBColor(0x99, 0x99, 0x99), italic=True)

        # ── Data Provenance ───────────────────────────────────────────
        doc.add_paragraph()
        _add_rule(doc)
        _section_header(doc, "A1", "Data Provenance & Disclaimer")
        _para(doc,
              "MRPL_PUBLIC: Sourced from public MRPL documentation.\n"
              "SYNTHETIC_DEMO: Synthetic data created for SOVEREIGN prototype demonstration.\n"
              "USER_UPLOAD: Files uploaded by the operator for this mission.\n\n"
              "IMPORTANT DISCLAIMER: This report was generated by SOVEREIGN, a sovereign on-premise "
              "AI prototype (MRPL Project 26117). Synthetic SOPs, thresholds, and equipment data used "
              "in demonstration cases are NOT actual MRPL confidential documents or real engineering "
              "limits. All AI findings require qualified human review and approval before any "
              "operational action is taken. SOVEREIGN does NOT replace human engineering judgment.",
              size=8)

        # ── Footer ────────────────────────────────────────────────────
        _add_rule(doc)
        footer_p = doc.add_paragraph(
            f"SOVEREIGN Industrial AI Workbench  |  MRPL  |  "
            f"Mission: {mission_id}  |  Generated: {now_str}  |  "
            f"All processing: LOCAL — No data left the premises"
        )
        footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _first_run(footer_p).font.size = Pt(7)
        _first_run(footer_p).font.color.rgb = RGBColor(0x80, 0x90, 0x70)

        doc.save(str(output_path))
        return str(output_path)
