"""
EcoSonic AI - Report Generation Module
Generates downloadable reports in CSV and PDF formats.
"""
import pandas as pd
import io
import re
from datetime import datetime
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_all_recordings, get_seasonal_data, get_locations
from config import APP_TITLE, APP_VERSION


def generate_recordings_csv(recordings: List[Dict]) -> bytes:
    if not recordings:
        return b"No recordings found"
    df = pd.DataFrame(recordings)
    cols = ["filename","upload_date","recording_date","location","season",
            "duration_seconds","analysis_status","predicted_category","confidence_score","is_demo"]
    available = [c for c in cols if c in df.columns]
    buf = io.BytesIO()
    df[available].to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def generate_seasonal_csv(seasonal_data: List[Dict]) -> bytes:
    if not seasonal_data:
        return b"No seasonal data found"
    buf = io.BytesIO()
    pd.DataFrame(seasonal_data).to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def generate_full_report_csv() -> bytes:
    recordings = get_all_recordings()
    if not recordings:
        return b"No data available for report"
    df = pd.DataFrame(recordings)
    meta = [f"# {APP_TITLE} - Ecological Monitoring Report",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Version: {APP_VERSION}", f"# Total Recordings: {len(df)}", ""]
    buf = io.BytesIO()
    buf.write("\n".join(meta).encode("utf-8"))
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def _strip_emoji(text):
    """Remove emoji and non-latin characters for PDF compatibility."""
    if not isinstance(text, str):
        return str(text)
    # Remove emojis and other non-BMP characters
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # Remove remaining non-latin1 chars
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        return text.encode('latin-1', errors='ignore').decode('latin-1')


def generate_pdf_report(recordings=None, seasonal_data=None, habitat_analysis=None, expert_notes=""):
    from fpdf import FPDF

    class EcoReport(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(27, 94, 32)
            self.cell(0, 10, APP_TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, "Biodiversity Soundscape and Habitat Change Monitor", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)
            self.set_draw_color(76, 175, 80)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

        def section_title(self, title):
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(27, 94, 32)
            self.cell(0, 10, _strip_emoji(title), new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 230, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

        def body_text(self, text):
            self.set_font("Helvetica", "", 10)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 5, _strip_emoji(text))
            self.ln(2)

    pdf = EcoReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Application Version: {APP_VERSION}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    if recordings:
        pdf.section_title("Audio Recordings Summary")
        pdf.body_text(f"Total recordings analyzed: {len(recordings)}")
        pdf.set_font("Helvetica", "B", 9)
        widths = [50, 25, 30, 25, 30, 25]
        headers = ["Filename", "Date", "Location", "Season", "Category", "Confidence"]
        for w, h in zip(widths, headers):
            pdf.cell(w, 7, h, border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for rec in recordings[:30]:
            row = [str(rec.get("filename",""))[:25], str(rec.get("recording_date",""))[:10],
                   str(rec.get("location",""))[:15], str(rec.get("season","")),
                   str(rec.get("predicted_category","")),
                   f"{rec.get('confidence_score',0):.0%}" if rec.get("confidence_score") else "N/A"]
            for w, val in zip(widths, row):
                pdf.cell(w, 6, val, border=1)
            pdf.ln()
        if len(recordings) > 30:
            pdf.body_text(f"... and {len(recordings) - 30} more recordings.")
        pdf.ln(5)

    if seasonal_data:
        pdf.section_title("Seasonal Biodiversity Comparison")
        for entry in seasonal_data[:20]:
            line = (f"Location: {entry.get('location','N/A')} | Season: {entry.get('season','N/A')} | "
                    f"Birds: {entry.get('bird_count',0)} | Frogs: {entry.get('frog_count',0)} | "
                    f"Insects: {entry.get('insect_count',0)}")
            pdf.body_text(line)
        pdf.ln(3)

    if habitat_analysis:
        pdf.section_title("Habitat Change Indicators")
        pdf.body_text(f"Overall Status: {habitat_analysis.get('status_label','N/A')}")
        indicators = habitat_analysis.get("indicators", [])
        if indicators:
            for ind in indicators:
                pdf.body_text(f"- {ind.get('category','')}: {ind.get('message','')}")
        else:
            pdf.body_text("No significant changes detected in the analyzed period.")
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 50, 50)
        pdf.multi_cell(0, 4, _strip_emoji(
            "DISCLAIMER: Changes in wildlife sound patterns can be caused by many factors "
            "including weather, season, time of day, recording equipment, and background noise. "
            "Reduced sound activity alone does not prove habitat destruction or biodiversity loss. "
            "Professional ecological assessment is required for definitive conclusions."))
        pdf.ln(3)

    if expert_notes:
        pdf.section_title("Expert Review Notes")
        pdf.body_text(expert_notes)

    return pdf.output()
