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
import io
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"


def clean_tex_inline(text: str) -> str:
    s = text
    reps = {
        r"\rightarrow": "→", r"\implies": "⇒", r"\times": "×",
        r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
        r"\neq": "≠", r"\infty": "∞", r"\uparrow": "↑", r"\downarrow": "↓",
        r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\approx": "≈",
        r"\pm": "±", r"\cdot": "·", r"\circ": "°", r"\sum": "∑", r"\prod": "∏",
        r"\sigma": "σ", r"\mu": "μ", r"\omega": "ω", r"\eta": "η",
        r"\gamma": "γ", r"\theta": "θ", r"\lambda": "λ", r"\pi": "π",
        r"\Delta": "Δ", r"\Omega": "Ω", r"\partial": "∂", r"\mathbb{R}": "ℝ",
        r"\mathbb{I}": "I", r"\dots": "...", r"\cdots": "...",
        r"\{": "{", r"\}": "}",
    }
    for k, v in reps.items():
        s = s.replace(k, v)
    s = re.sub(r"\\(text|mathrm|mathbf|mathcal|mathbb)\{([^}]+)\}", r"\2", s)
    s = re.sub(r"\\(hat|bar|tilde)\{([^}]+)\}", r"\2̂", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"√(\1)", s)
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    return s


def clean_tex_for_mathtext(s: str) -> str:
    s = s.strip()
    s = s.replace(r"\underbrace", "")
    s = re.sub(r"\\le\b", r"\\leq", s)
    s = re.sub(r"\\ge\b", r"\\geq", s)
    s = re.sub(r"_\{([^}]+)\\text\{ times\}\}", r"_{\1}", s)
    s = re.sub(r"_\{([^}]+)\\mathrm\{ times\}\}", r"_{\1}", s)
    s = s.replace(r"\mathbb{I}", r"\mathbf{I}")
    s = s.replace(r"\mathbb{R}", r"\mathrm{R}")
    s = s.replace(r"\mathcal{R}e", r"\mathrm{Re}")
    s = s.replace(r"\text", r"\mathrm")
    s = s.replace(r"\%", "%")
    for macro in [r"\left", r"\right", r"\Big", r"\big", r"\!", r"\,", r"\;"]:
        s = s.replace(macro, "")
    return s


def render_math_to_png(tex: str, fontsize: float = 12.5) -> io.BytesIO:
    cleaned = clean_tex_for_mathtext(tex)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${cleaned}$", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def add_display_equation(doc, raw_eq: str):
    raw = raw_eq.strip("$").strip()
    m_eq = re.search(r"\\qquad\s*\((\d+)\)\s*$", raw)
    eq_num = m_eq.group(1) if m_eq else None
    if m_eq:
        raw = raw[: m_eq.start()].strip()

    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in tbl.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = OxmlElement("w:tcBorders")
        for b in ["top", "left", "bottom", "right"]:
            edge = OxmlElement(f"w:{b}")
            edge.set(qn("w:val"), "none")
            tcBorders.append(edge)
        tcPr.append(tcBorders)

    c0 = tbl.cell(0, 0)
    c1 = tbl.cell(0, 1)
    c0.width = Inches(5.8)
    c1.width = Inches(0.7)

    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if r"\begin{bmatrix}" in raw:
        clean_text = raw.replace(r"\begin{bmatrix}", "[").replace(r"\end{bmatrix}", "]")
        clean_text = clean_text.replace(r"\\", " ; ").replace("&", " ")
        clean_text = clean_tex_inline(clean_text)
        r0 = p0.add_run(clean_text)
        r0.italic = True
        r0.font.size = Pt(11)
    else:
        try:
            buf = render_math_to_png(raw)
            p0.add_run().add_picture(buf)
        except Exception:
            clean_text = clean_tex_inline(raw)
            r0 = p0.add_run(clean_text)
            r0.italic = True
            r0.font.size = Pt(11)

    if eq_num:
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r1 = p1.add_run(f"({eq_num})")
        r1.font.size = Pt(11)


def set_cell_background(cell, fill_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def add_formatted_runs(paragraph, text: str, base_font_size: int = 11):
    """Parse markdown **bold** and `code` spans into formatted Word runs."""
    cleaned = clean_tex_inline(text)

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
        # Split on '|' only outside of $...$ math and `...` code spans
        raw_cells, buf, in_math, in_code = [], "", False, False
        for ch in stripped:
            if ch == "`" and not in_math:
                in_code = not in_code
            elif ch == "$" and not in_code:
                in_math = not in_math
            if ch == "|" and not in_math and not in_code:
                raw_cells.append(buf)
                buf = ""
            else:
                buf += ch
        raw_cells.append(buf)
        cells = [c.strip().replace("<br>", "\n") for c in raw_cells]
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

        # Author Grid HTML
        if "<div class=\"author-grid\">" in stripped or "<div class='author-grid'>" in stripped:
            auth_lines = [stripped]
            i += 1
            while i < n and "</div>\n</div>" not in lines[i] and not lines[i].strip().startswith("##"):
                auth_lines.append(lines[i])
                if lines[i].strip() == "</div>":
                    i += 1
                    break
                i += 1
            full_auth_html = "\n".join(auth_lines)
            authors = re.findall(r"<b>(.*?)</b><br/?>\s*([A-Za-z0-9]+)", full_auth_html)
            if authors:
                tbl_a = doc.add_table(rows=1, cols=len(authors))
                tbl_a.alignment = WD_TABLE_ALIGNMENT.CENTER
                for a_idx, (name, sid) in enumerate(authors):
                    c = tbl_a.cell(0, a_idx)
                    # remove borders
                    tcPr = c._tc.get_or_add_tcPr()
                    tcBorders = OxmlElement("w:tcBorders")
                    for b in ["top", "left", "bottom", "right"]:
                        edge = OxmlElement(f"w:{b}")
                        edge.set(qn("w:val"), "none")
                        tcBorders.append(edge)
                    tcPr.append(tcBorders)
                    p = c.paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r_name = p.add_run(name + "\n")
                    r_name.bold = True
                    r_name.font.size = Pt(10)
                    r_sid = p.add_run(sid.strip())
                    r_sid.font.size = Pt(9.5)
                    r_sid.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
                doc.add_paragraph()
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
            add_display_equation(doc, stripped)
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
            if "BÁO CÁO ĐỒ ÁN MÔN HỌC" in title_text or "COURSE PROJECT REPORT" in title_text:
                p_top = doc.add_paragraph()
                p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r_top = p_top.add_run(clean_tex_inline(title_text))
                r_top.bold = True
                r_top.font.name = "Times New Roman"
                r_top.font.size = Pt(15.5)
                r_top.font.color.rgb = RGBColor(0x12, 0x3E, 0x67)

                p_line = doc.add_paragraph()
                p_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r_line = p_line.add_run("―" * 46)
                r_line.font.color.rgb = RGBColor(0x12, 0x3E, 0x67)
                r_line.bold = True
            elif "ĐỀ TÀI:" in title_text or "TOPIC:" in title_text:
                p_topic = doc.add_paragraph()
                p_topic.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r_topic = p_topic.add_run(clean_tex_inline(title_text))
                r_topic.bold = True
                r_topic.font.name = "Times New Roman"
                r_topic.font.size = Pt(12.5)
                r_topic.font.color.rgb = RGBColor(0x12, 0x3E, 0x67)
            else:
                h = doc.add_heading("", level=level if level > 1 else 0)
                add_formatted_runs(h, title_text, base_font_size=15 if level == 1 else (13 if level == 2 else 11.5))
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
                    prefix = "Hình" if "_VI" in os.path.basename(md_path) else "Figure"
                    r_cap = p_cap.add_run(f"{prefix}: {caption}")
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
        "awgn": "Additive White Gaussian Noise (sigma=10.0)",
        "salt_pepper": "Impulse Salt-and-Pepper Noise (pa=0.02, pb=0.02)",
        "rayleigh": "Skewed Rayleigh Low-Light Noise (a=-8, b=160)",
        "periodic": "2D Sinusoidal Periodic Interference (A=24, u0=24, v0=18)",
        "motion_blur": "Linear Motion Blur (L=7px, theta=12deg) + AWGN (sigma=7.0)",
    }
    rest_methods = {
        "clean": "Identity (No Restoration Needed)",
        "awgn": "2D Haar Wavelet Soft-Threshold Denoising + Unsharp Highboost",
        "salt_pepper": "3x3 Center-Weighted Median Filter",
        "rayleigh": "3x3 Harmonic Mean Filter + Unsharp Highboost",
        "periodic": "2D Frequency-Domain Butterworth Notch Reject Filter",
        "motion_blur": "2D Haar Wavelet Shrinkage + Regularized Wiener / CLS Deconvolution",
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
    # 1. Build English / Bilingual Word Report
    build_docx_from_markdown("report/FINAL_REPORT.md", "report/FINAL_REPORT_IMP302.docx")
    # 2. Build Vietnamese Word Report
    build_docx_from_markdown("report/FINAL_REPORT_VI.md", "report/FINAL_REPORT_VI_IMP302.docx")
    # 3. Export Manifest without any slide references
    build_expanded_manifest_csv()

