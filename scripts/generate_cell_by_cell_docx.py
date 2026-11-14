"""
TrajPilot CityFlow | Universal Jupyter Notebook Cell-by-Cell Explanation & DOCX Generator
========================================================================================
Generates a comprehensive, professional Microsoft Word (.docx) technical report that
documents all 172 cells of TrajPilot_Research_20261005.ipynb individually with:
- Exact original code & markdown reproduction
- Live stdout / tabular outputs
- Embedded high-resolution PNG figures & captions
- Continuous ~100-word paragraph explanations (no bullet points)
- Technical output evaluations with exact numbers & metrics
- Visualization analyses (axes, units, legends, trends, conclusions)
- Section connections between all 8 research phases
- Final Notebook Summary & Comprehensive 172-Row Cell Index Table
"""

import os
import sys
import json
import base64
import io
from pathlib import Path
import nbformat
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOK_PATH = PROJECT_ROOT / "TrajPilot_Research_20261005.ipynb"
OUTPUT_DOCX_PATH = PROJECT_ROOT / "TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx"

# Color Palette Constants
COLOR_NAVY = "1B365D"       # Primary Headings & Accents
COLOR_BLUE = "2B6CB0"       # Secondary Headings & Highlights
COLOR_CHARCOAL = "1F2937"   # Main Body Text
COLOR_CODE_BG = "F8FAFC"    # Light Gray for Code Blocks
COLOR_CODE_BORDER = "CBD5E1"# Soft Gray Border for Code
COLOR_OUT_BG = "F1F5F9"     # Light Slate for Output Blocks
COLOR_OUT_BORDER = "94A3B8" # Slate Border for Output
COLOR_CALLOUT_BG = "EFF6FF" # Light Blue for Section Connections
COLOR_CALLOUT_BORDER = "3B82F6"

def set_cell_background(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_cell_borders(cell, border_color="CBD5E1", left_color="2B6CB0", left_size=18, other_size=4):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{other_size}" w:space="0" w:color="{border_color}"/>'
        f'<w:left w:val="single" w:sz="{left_size}" w:space="0" w:color="{left_color}"/>'
        f'<w:bottom w:val="single" w:sz="{other_size}" w:space="0" w:color="{border_color}"/>'
        f'<w:right w:val="single" w:sz="{other_size}" w:space="0" w:color="{border_color}"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

def add_styled_box(doc, text, bg_hex, border_hex, left_color, is_code=False, max_lines=60):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    cell = tbl.cell(0, 0)
    
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
    set_cell_borders(cell, border_color=border_hex, left_color=left_color, left_size=18, other_size=4)
    
    lines = text.splitlines()
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... [{len(lines) - max_lines} additional lines truncated for document layout] ..."]
    clean_text = "\n".join(lines) if lines else "(empty)"
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(clean_text)
    run.font.name = "Consolas" if is_code else "Calibri"
    run.font.size = Pt(8.5 if is_code else 9.5)
    run.font.color.rgb = RGBColor(15, 23, 42)
    
    # spacing after table
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def add_paragraph_explanation(doc, heading_text, paragraph_text, heading_color="1B365D"):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(2)
    h.paragraph_format.keep_with_next = True
    r_h = h.add_run(heading_text)
    r_h.font.name = "Calibri"
    r_h.font.size = Pt(11)
    r_h.font.bold = True
    r_h.font.color.rgb = RGBColor(int(heading_color[:2], 16), int(heading_color[2:4], 16), int(heading_color[4:], 16))
    
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    r_p = p.add_run(paragraph_text)
    r_p.font.name = "Calibri"
    r_p.font.size = Pt(10.5)
    r_p.font.color.rgb = RGBColor(31, 41, 55)

def add_callout(doc, title, text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    cell = tbl.cell(0, 0)
    
    set_cell_background(cell, COLOR_CALLOUT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    set_cell_borders(cell, border_color=COLOR_CALLOUT_BORDER, left_color=COLOR_CALLOUT_BORDER, left_size=24, other_size=6)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    r_t = p.add_run(title.upper() + "\n")
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(11)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(30, 64, 175)
    
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(10)
    r_body.font.color.rgb = RGBColor(30, 58, 138)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(8)

print("Helper functions compiled.")
