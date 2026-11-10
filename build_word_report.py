"""
Build a Professionally Formatted Microsoft Word Report (`report/FINAL_REPORT_IMP302.docx`)
and Expanded Dataset Manifest CSV (`data/splits/expanded_dataset_manifest.csv`)
==========================================================================================
Converts `report/FINAL_REPORT.md` into a clean academic `.docx` document with:
- Proper Word Tables (`Table Grid` style, bold header rows, shaded header background)
- Styled Headings (Heading 1, Heading 2, Heading 3)
- Rich inline bold/italic/code formatting
- Centered high-resolution embedded figures (`fig01`..`fig10`)
- Also exports `data/splits/expanded_dataset_manifest.csv` listing all 690 expanded plate records.
"""

import os
import re
import csv
import json
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def add_formatted_runs(paragraph, text: str, base_font_size: int = 11):
    """Parse markdown **bold** and `code` spans into formatted Word runs."""
    # Clean LaTeX inline math markers for cleaner Word readability
    cleaned = text.replace("\\rightarrow", "→").replace("\\implies", "⇒").replace("\\times", "×")
    cleaned = cleaned.replace("\\le", "≤").replace("\\ge", "≥").replace("\\neq", "≠")
    cleaned = cleaned.replace("\\infty", "∞").replace("\\uparrow", "↑").replace("\\downarrow", "↓")
    cleaned = re.sub(r"\$([^$]+)\$", r"\1", cleaned)

    tokens = re.split(r"(\*\*.*?\*\*|`.*?`)", cleaned)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            run = paragraph.add_run(tok[2:-2])
            run.bold = True
            run.font.size = Pt(base_font_size)
        elif tok.startswith("`") and tok.endswith("`"):
            run = paragraph.add_run(tok[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(base_font_size - 0.5)
            run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        else:
            run = paragraph.add_run(tok)
            run.font.size = Pt(base_font_size)


def add_markdown_table(doc, table_lines: list[str]):
    rows_data = []
    for line in table_lines:
        stripped = line.strip().strip("|")
        cells = [c.strip().replace("<br>", "\n") for c in stripped.split("|")]
        # Skip separator line like | :--- | :---: |
        if all(re.match(r"^[:\-]+$", c.replace(" ", "")) for c in cells if c):
            continue
        rows_data.append(cells)

    if not rows_data:
        return

    n_rows = len(rows_data)
    n_cols = max(len(r) for r in rows_data)
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row in enumerate(rows_data):
        for c_idx in range(n_cols):
            val = row[c_idx] if c_idx < len(row) else ""
            cell = table.cell(r_idx, c_idx)
            cell.text = ""
            p = cell.paragraphs[0]
            add_formatted_runs(p, val, base_font_size=9.5 if r_idx > 0 else 10)
            if r_idx == 0:
                set_cell_background(cell, "D9E2F3")
                for run in p.runs:
                    run.bold = True
    doc.add_paragraph()


def build_docx_from_markdown(md_path: str = "report/FINAL_REPORT.md", out_docx: str = "report/FINAL_REPORT_IMP302.docx"):
    doc = docx.Document()

    # Configure default Normal style
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11.5)

    lines = open(md_path, encoding="utf-8").read().splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped or stripped == "---":
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(code_lines))
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
            continue

        # Display math block $$...$$
        if stripped.startswith("$$"):
            math_text = stripped.strip("$").strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(math_text)
            r.italic = True
            r.font.size = Pt(11)
            i += 1
            continue

        # Markdown Table
        if stripped.startswith("|") and stripped.endswith("|"):
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            add_markdown_table(doc, tbl_lines)
            continue

        # Headings
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 4)
            title_text = stripped.lstrip("#").strip()
            h = doc.add_heading("", level=level if level > 1 else 0)
            add_formatted_runs(h, title_text, base_font_size=16 if level == 1 else (14 if level == 2 else 12))
            i += 1
            continue

        # Image ![alt](path)
        if stripped.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                caption, rel_path = m.group(1), m.group(2)
                img_path = os.path.normpath(os.path.join(os.path.dirname(md_path), rel_path))
                if os.path.exists(img_path):
                    p_img = doc.add_paragraph()
                    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p_img.add_run().add_picture(img_path, width=Inches(6.3))
                    p_cap = doc.add_paragraph()
                    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r_cap = p_cap.add_run(f"Hình: {caption}")
                    r_cap.italic = True
                    r_cap.font.size = Pt(10)
            i += 1
            continue

        # Bullet or numbered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            add_formatted_runs(p, stripped[2:].strip(), base_font_size=11)
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            content = re.sub(r"^\d+\.\s+", "", stripped)
            p = doc.add_paragraph(style="List Number")
            add_formatted_runs(p, content, base_font_size=11)
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            add_formatted_runs(p, stripped.lstrip("> ").strip(), base_font_size=11)
            i += 1
            continue

        # Standard paragraph
        p = doc.add_paragraph()
        add_formatted_runs(p, stripped, base_font_size=11.5)
        i += 1

    doc.save(out_docx)
    print(f"Successfully generated formatted Word report: {out_docx}")


def build_expanded_manifest_csv():
    with open("data/splits/dataset_splits.json", encoding="utf-8") as f:
        meta = json.load(f)

    deg_descriptions = {
        "clean": "Original Clean Camera Image (Reference)",
        "awgn": "Additive White Gaussian Noise (sigma=10.0, Doc Ch. 3.2 Slide 7)",
        "salt_pepper": "Impulse Salt-and-Pepper Noise (pa=0.02, pb=0.02, Doc Ch. 3.2 Slide 12)",
        "rayleigh": "Skewed Rayleigh Low-Light Noise (a=-8, b=160, Doc Ch. 3.2 Slide 8)",
        "periodic": "2D Sinusoidal Periodic Interference (A=24, u0=24, v0=18, Doc Ch. 3.3 Slides 18-22)",
        "motion_blur": "Linear Motion Blur (L=7px, theta=12deg) + AWGN (sigma=7.0, Doc Ch. 3.4 & 3.6-3.7)",
    }
    rest_methods = {
        "clean": "Identity (No Restoration Needed)",
        "awgn": "2D Haar Wavelet Soft-Threshold Denoising (Ch. 3.5) + Unsharp Highboost (Ch. 2.4)",
        "salt_pepper": "3x3 Center-Weighted Median Filter (Ch. 2.2 Slide 13 & Ch. 3.3)",
        "rayleigh": "3x3 Harmonic Mean Filter (Ch. 3.3 Slide 15) + Unsharp Highboost (Ch. 2.4)",
        "periodic": "2D Frequency-Domain Butterworth Notch Reject Filter (Ch. 3.3 Slides 18-22)",
        "motion_blur": "2D Haar Wavelet Shrinkage + Regularized Wiener / CLS Deconvolution (Ch. 3.6-3.7)",
    }

    csv_path = "data/splits/expanded_dataset_manifest.csv"
    rows = []
    idx = 1
    for split_name in ["train", "val", "test"]:
        for rec in meta["splits"][split_name]:
            for deg in meta["degradation_types"]:
                rows.append({
                    "sample_id": idx,
                    "plate_base": rec["base"],
                    "split": split_name,
                    "degradation_type": deg,
                    "gt_label": rec["gt_label"],
                    "num_chars": len(rec["gt_label"]),
                    "source_img_path": rec["img_path"],
                    "degradation_model_doc_ref": deg_descriptions[deg],
                    "restoration_filter_doc_ref": rest_methods[deg],
                })
                idx += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully exported {len(rows)} expanded plate records to {csv_path}")


if __name__ == "__main__":
    build_docx_from_markdown()
    build_expanded_manifest_csv()

