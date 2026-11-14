"""
Comprehensive DOCX Generation Pipeline for TrajPilot_Research_20261005.ipynb
===========================================================================
Produces TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx containing
complete cell-by-cell explanations, original code, stdout outputs, embedded
high-resolution PNG visualizations, section connections, final summary, and
a 172-row complete cell index.
"""

import os
import sys
import json
import base64
import io
import time
from pathlib import Path
import nbformat
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path("c:/Users/SOMUJIT/Desktop/cityflow").resolve()
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
NOTEBOOK_PATH = PROJECT_ROOT / "TrajPilot_Research_20261005.ipynb"
OUTPUT_DOCX_PATH = PROJECT_ROOT / "TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx"

# Color Palette Constants
COLOR_NAVY = "1B365D"       # Deep Navy
COLOR_BLUE = "2B6CB0"       # Slate Blue
COLOR_DARK = "1F2937"       # Charcoal Body Text
COLOR_CODE_BG = "F8FAFC"    # Light Gray
COLOR_CODE_BORDER = "CBD5E1"# Soft Gray
COLOR_OUT_BG = "F1F5F9"     # Light Slate
COLOR_OUT_BORDER = "94A3B8" # Slate Gray
COLOR_CALLOUT_BG = "EFF6FF" # Light Blue
COLOR_CALLOUT_BORDER = "3B82F6"

def set_cell_background(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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

def add_styled_box(doc, text, bg_hex, border_hex, left_color, is_code=False, max_lines=75):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    cell = tbl.cell(0, 0)
    
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
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
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def add_paragraph_explanation(doc, heading_text, paragraph_text, heading_color=COLOR_NAVY):
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
    set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
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

def add_figure_block(doc, image_path_or_bytes, caption, explanation_text):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    
    run_img = p_img.add_run()
    if isinstance(image_path_or_bytes, (str, Path)):
        run_img.add_picture(str(image_path_or_bytes), width=Inches(5.8))
    else:
        run_img.add_picture(io.BytesIO(image_path_or_bytes), width=Inches(5.8))
        
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(0)
    p_cap.paragraph_format.space_after = Pt(6)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Calibri"
    r_cap.font.size = Pt(9.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(71, 85, 105)
    
    add_paragraph_explanation(doc, "VISUALIZATION EXPLANATION", explanation_text, heading_color=COLOR_BLUE)

print("Core builder logic loaded.")
