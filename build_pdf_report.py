"""
Export `report/FINAL_REPORT.md` Preview to a High-Resolution Academic PDF (`report/FINAL_REPORT.pdf`)
=====================================================================================================
Converts `report/FINAL_REPORT.md` into:
1. `report/FINAL_REPORT.html` (Self-contained GitHub-style Markdown preview with embedded base64 images,
   styled academic tables, blockquotes, and cleanly rendered mathematical formulas).
2. `report/FINAL_REPORT.pdf` & `report/FINAL_REPORT_IMP302.pdf` (A4 PDF rendered via Headless Edge/Chrome
   or PyMuPDF `fitz.Story` fallback).
"""

import os
import re
import html
import base64
import shutil
import subprocess
import pymupdf as fitz


def find_matching_brace(s: str, open_idx: int) -> int:
    """Find index of matching '}' for '{' at open_idx."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def replace_two_arg_macro(s: str, macro: str, formatter) -> str:
    """Safely replace \macro{arg1}{arg2} supporting arbitrary nested braces."""
    while True:
        idx = s.find(macro)
        if idx == -1:
            break
        p1 = idx + len(macro)
        while p1 < len(s) and s[p1] == " ":
            p1 += 1
        if p1 >= len(s) or s[p1] != "{":
            break
        q1 = find_matching_brace(s, p1)
        if q1 == -1:
            break
        p2 = q1 + 1
        while p2 < len(s) and s[p2] == " ":
            p2 += 1
        if p2 >= len(s) or s[p2] != "{":
            break
        q2 = find_matching_brace(s, p2)
        if q2 == -1:
            break
        arg1 = s[p1 + 1 : q1]
        arg2 = s[p2 + 1 : q2]
        s = s[:idx] + formatter(arg1, arg2) + s[q2 + 1 :]
    return s


def replace_one_arg_macro(s: str, macro: str, formatter) -> str:
    """Safely replace \macro{arg1} supporting arbitrary nested braces."""
    while True:
        idx = s.find(macro)
        if idx == -1:
            break
        p1 = idx + len(macro)
        while p1 < len(s) and s[p1] == " ":
            p1 += 1
        if p1 >= len(s) or s[p1] != "{":
            break
        q1 = find_matching_brace(s, p1)
        if q1 == -1:
            break
        arg1 = s[p1 + 1 : q1]
        s = s[:idx] + formatter(arg1) + s[q1 + 1 :]
    return s


def latex_to_html_math(tex: str, is_display: bool = False) -> str:
    """
    Convert LaTeX math expressions from FINAL_REPORT.md into clean, publication-grade
    HTML math with proper subscripts, superscripts, fractions, matrices, and Greek symbols.
    """
    s = tex.strip()

    # 1. Handle \begin{bmatrix} ... \end{bmatrix}
    def _repl_bmatrix(m):
        body = m.group(1).strip()
        rows = [r.strip() for r in body.split("\\\\") if r.strip()]
        row_html = []
        for r in rows:
            cols = [c.strip() for c in r.split("&")]
            tds = "".join(f"<td>{c}</td>" for c in cols)
            row_html.append(f"<tr>{tds}</tr>")
        return f"<table class='bmatrix'>{''.join(row_html)}</table>"

    s = re.sub(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", _repl_bmatrix, s, flags=re.DOTALL)

    # 2. Clean sizing & spacing macros
    s = s.replace(r"\quad", " &emsp; ").replace(r"\qquad", " &emsp;&emsp; ")
    for macro in [r"\left", r"\right", r"\Big", r"\big", r"\!", r"\,"]:
        s = s.replace(macro, "")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\{", "&#123;").replace(r"\}", "&#125;")

    # 3. Handle \underbrace{A}_{B}
    while True:
        idx = s.find(r"\underbrace")
        if idx == -1:
            break
        p1 = idx + len(r"\underbrace")
        if p1 >= len(s) or s[p1] != "{":
            break
        q1 = find_matching_brace(s, p1)
        if q1 == -1:
            break
        rest = s[q1 + 1 :]
        if rest.startswith("_{"):
            q2 = find_matching_brace(s, q1 + 2)
            if q2 != -1:
                top_expr = s[p1 + 1 : q1]
                sub_expr = s[q1 + 3 : q2]
                repl = (
                    f"<span style='display:inline-block;text-align:center;vertical-align:middle;'>"
                    f"<span style='display:block;border-bottom:1px solid #444;padding:0 3px;'>{top_expr}</span>"
                    f"<span style='display:block;font-size:0.75em;color:#444;'>{sub_expr}</span></span>"
                )
                s = s[:idx] + repl + s[q2 + 1 :]
                continue
        break

    # 4. Handle \stackrel{top}{bot} and \frac{num}{den} and \sqrt{arg}
    for _ in range(3):
        s = replace_two_arg_macro(
            s,
            r"\stackrel",
            lambda a, b: (
                f"<span style='display:inline-block;text-align:center;vertical-align:middle;line-height:1.0;'>"
                f"<span style='display:block;font-size:0.72em;'>{a}</span>"
                f"<span style='display:block;'>{b}</span></span>"
            ),
        )
        s = replace_two_arg_macro(
            s,
            r"\frac",
            lambda a, b: (
                f"<span class='frac'>"
                f"<span class='num'>{a}</span>"
                f"<span class='den'>{b}</span></span>"
            ),
        )
        s = replace_one_arg_macro(
            s,
            r"\sqrt",
            lambda a: f"&radic;<span style='border-top:1px solid #222;padding-top:1px;'>{a}</span>",
        )
        s = replace_one_arg_macro(
            s,
            r"\text",
            lambda a: f"<span style='font-style:normal;font-family:Segoe UI,Arial,sans-serif;'>{a}</span>",
        )
        s = replace_one_arg_macro(s, r"\mathbf", lambda a: f"<b>{a}</b>")
        s = replace_one_arg_macro(s, r"\mathcal", lambda a: f"<i><b>{a}</b></i>")
        s = replace_one_arg_macro(s, r"\mathbb", lambda a: f"<b>{a}</b>")
        s = replace_one_arg_macro(s, r"\hat", lambda a: f"{a}&#770;")
        s = replace_one_arg_macro(s, r"\bar", lambda a: f"{a}&#772;")
        s = replace_one_arg_macro(s, r"\tilde", lambda a: f"{a}&#771;")

    # 5. Greek & Math symbols
    replacements = {
        r"\alpha": "&alpha;",
        r"\beta": "&beta;",
        r"\gamma": "&gamma;",
        r"\theta": "&theta;",
        r"\lambda": "&lambda;",
        r"\mu": "&mu;",
        r"\pi": "&pi;",
        r"\sigma": "&sigma;",
        r"\omega": "&omega;",
        r"\eta": "&eta;",
        r"\epsilon": "&epsilon;",
        r"\phi": "&phi;",
        r"\psi": "&psi;",
        r"\partial": "&part;",
        r"\Delta": "&Delta;",
        r"\Omega": "&Omega;",
        r"\infty": "&infin;",
        r"\sum": "&sum;",
        r"\prod": "&prod;",
        r"\int": "&int;",
        r"\in": "&isin;",
        r"\notin": "&notin;",
        r"\cup": "&cup;",
        r"\cap": "&cap;",
        r"\subset": "&sub;",
        r"\le": "&le;",
        r"\ge": "&ge;",
        r"\neq": "&ne;",
        r"\approx": "&approx;",
        r"\equiv": "&equiv;",
        r"\sim": "&sim;",
        r"\pm": "&plusmn;",
        r"\mp": "&#8723;",
        r"\times": "&times;",
        r"\cdot": "&middot;",
        r"\bullet": "&bull;",
        r"\circ": "&deg;",
        r"\rightarrow": "&rarr;",
        r"\leftarrow": "&larr;",
        r"\Longleftrightarrow": "&hArr;",
        r"\implies": "&rArr;",
        r"\Rightarrow": "&rArr;",
        r"\uparrow": "&uarr;",
        r"\downarrow": "&darr;",
        r"\ll": "&#8810;",
        r"\gg": "&#8811;",
        r"\dots": "&hellip;",
        r"\cdots": "&ctdot;",
        r"\|": "&#8214;",
        r"\ln": "<span style='font-style:normal;'>ln</span>",
        r"\log": "<span style='font-style:normal;'>log</span>",
        r"\sin": "<span style='font-style:normal;'>sin</span>",
        r"\cos": "<span style='font-style:normal;'>cos</span>",
        r"\exp": "<span style='font-style:normal;'>exp</span>",
        r"\max": "<span style='font-style:normal;'>max</span>",
        r"\min": "<span style='font-style:normal;'>min</span>",
        r"\arg": "<span style='font-style:normal;'>arg</span>",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # 6. Subscripts and superscripts: _{...}, ^{...}, _x, ^x
    for _ in range(3):
        s = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([a-zA-Z0-9\*])", r"<sup>\1</sup>", s)

    if is_display:
        return f"<div class='math-block'>{s}</div>"
    return f"<span class='math-inline'>{s}</span>"


def format_inline_markdown(text: str) -> str:
    """Convert inline markdown (links, code spans, math, bold, italic) to HTML."""
    code_placeholders = {}
    def _save_code(m):
        key = f"@@CODE{len(code_placeholders)}@@"
        code_placeholders[key] = f"<code>{html.escape(m.group(1))}</code>"
        return key
    text = re.sub(r"`([^`]+)`", _save_code, text)

    math_placeholders = {}
    def _save_math(m):
        key = f"@@MATH{len(math_placeholders)}@@"
        math_placeholders[key] = latex_to_html_math(m.group(1), is_display=False)
        return key
    text = re.sub(r"\$([^$]+)\$", _save_math, text)

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    for k, v in math_placeholders.items():
        text = text.replace(k, v)
    for k, v in code_placeholders.items():
        text = text.replace(k, v)
    return text


def markdown_to_html_document(md_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(md_path))
    lines = open(md_path, encoding="utf-8").read().splitlines()

    html_parts = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            html_parts.append("<hr/>")
            i += 1
            continue

        # Code block ```
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            escaped_code = html.escape("\n".join(code_lines))
            html_parts.append(f"<pre class='code-block'><code class='lang-{lang}'>{escaped_code}</code></pre>")
            continue

        # Display math $$...$$
        if stripped.startswith("$$"):
            math_lines = []
            if stripped.endswith("$$") and len(stripped) > 4:
                math_lines.append(stripped[2:-2])
                i += 1
            else:
                math_lines.append(stripped[2:])
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i])
                    i += 1
                if i < n:
                    math_lines.append(lines[i].strip()[:-2])
                    i += 1
            html_parts.append(latex_to_html_math(" ".join(math_lines), is_display=True))
            continue

        # Markdown Table
        if stripped.startswith("|") and stripped.endswith("|"):
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            rows_html = []
            is_header = True
            for tline in tbl_lines:
                s_row = tline.strip().strip("|")
                cells = [c.strip() for c in s_row.split("|")]
                if all(re.match(r"^[:\-]+$", c.replace(" ", "")) for c in cells if c):
                    continue
                tag = "th" if is_header else "td"
                row_cells = "".join(f"<{tag}>{format_inline_markdown(c)}</{tag}>" for c in cells)
                rows_html.append(f"<tr>{row_cells}</tr>")
                is_header = False
            html_parts.append("<div class='table-wrap'><table class='data-table'>" + "".join(rows_html) + "</table></div>")
            continue

        # Headings
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 4)
            title = format_inline_markdown(stripped.lstrip("#").strip())
            html_parts.append(f"<h{level}>{title}</h{level}>")
            i += 1
            continue

        # Image ![alt](path)
        if stripped.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                caption, rel_path = m.group(1), m.group(2)
                img_abs = os.path.normpath(os.path.join(base_dir, rel_path))
                if os.path.exists(img_abs):
                    with open(img_abs, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    html_parts.append(
                        f"<figure class='report-fig'>"
                        f"<img src='data:image/png;base64,{b64}' alt='{html.escape(caption)}'/>"
                        f"<figcaption><b>Hình minh họa:</b> {format_inline_markdown(caption)}</figcaption>"
                        f"</figure>"
                    )
            i += 1
            continue

        # Blockquote >
        if stripped.startswith(">"):
            bq_lines = []
            while i < n and lines[i].strip().startswith(">"):
                bq_lines.append(format_inline_markdown(lines[i].strip().lstrip(">").strip()))
                i += 1
            html_parts.append("<blockquote>" + "<br/>".join(bq_lines) + "</blockquote>")
            continue

        # Unordered list - or *
        if stripped.startswith("- ") or stripped.startswith("* "):
            items = []
            while i < n and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                items.append(f"<li>{format_inline_markdown(lines[i].strip()[2:].strip())}</li>")
                i += 1
            html_parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        # Ordered list 1. (preserve explicit item number via value='k' even across math blocks)
        m_num = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m_num:
            items = []
            while i < n:
                m_cur = re.match(r"^(\d+)\.\s+(.*)", lines[i].strip())
                if not m_cur:
                    break
                num_val, item_txt = m_cur.group(1), m_cur.group(2)
                i += 1
                # Collect indented continuation or display math belonging to this numbered item
                sub_parts = [format_inline_markdown(item_txt)]
                while i < n and lines[i].strip().startswith("$$"):
                    s_math = lines[i].strip()
                    if s_math.endswith("$$") and len(s_math) > 4:
                        sub_parts.append(latex_to_html_math(s_math[2:-2], is_display=True))
                        i += 1
                    else:
                        break
                items.append(f"<li value='{num_val}'>{''.join(sub_parts)}</li>")
            html_parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        # Normal paragraph
        html_parts.append(f"<p>{format_inline_markdown(stripped)}</p>")
        i += 1

    css = """
    @page {
        size: A4 portrait;
        margin: 15mm 15mm 15mm 15mm;
    }
    body {
        font-family: 'Georgia', 'Times New Roman', 'Computer Modern Serif', serif;
        font-size: 10.3pt;
        line-height: 1.54;
        color: #141414;
        max-width: 960px;
        margin: 0 auto;
        padding: 8px 18px;
        background: #ffffff;
    }
    h1 {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 16.5pt;
        font-weight: 700;
        color: #111111;
        border-top: 3.5px solid #111111;
        border-bottom: 1.5px solid #111111;
        padding: 12px 6px;
        margin-top: 10px;
        margin-bottom: 14px;
        text-align: center;
        line-height: 1.32;
    }
    h2 {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 13pt;
        font-weight: 700;
        color: #8C1515;
        border-bottom: 1px solid #d0d7de;
        padding-bottom: 3px;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }
    h3 {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 11.2pt;
        font-weight: 700;
        color: #1f2328;
        margin-top: 13px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }
    h4 {
        font-family: 'Georgia', 'Times New Roman', serif;
        font-size: 10.3pt;
        font-weight: 700;
        font-style: italic;
        color: #333333;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }
    p, li {
        text-align: justify;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    ul, ol {
        margin-top: 4px;
        margin-bottom: 8px;
        padding-left: 22px;
    }
    hr {
        border: 0;
        border-top: 1px solid #c8d1d9;
        margin: 14px 0;
    }
    code {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 9pt;
        background-color: #f4f6f8;
        color: #8C1515;
        padding: 1px 4px;
        border-radius: 3px;
    }
    pre.code-block {
        background-color: #f8f9fa;
        border: 1px solid #d0d7de;
        border-radius: 4px;
        padding: 9px 12px;
        font-size: 8.6pt;
        line-height: 1.38;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    pre.code-block code {
        background: none;
        padding: 0;
        color: #1f2328;
    }
    .table-wrap {
        margin: 12px 0 15px 0;
        page-break-inside: avoid;
    }
    table.data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8.7pt;
        line-height: 1.36;
        border-top: 2px solid #111111;
        border-bottom: 2px solid #111111;
    }
    table.data-table th, table.data-table td {
        border-bottom: 1px solid #d8dee4;
        border-left: none;
        border-right: none;
        padding: 5px 7px;
        vertical-align: top;
        text-align: left;
    }
    table.data-table th {
        background-color: #f6f8fa;
        color: #111111;
        font-weight: 700;
        border-bottom: 1.5px solid #111111;
        text-align: left;
    }
    table.data-table tr:nth-child(even) td {
        background-color: #fcfcfd;
    }
    table.bmatrix {
        display: inline-table;
        width: auto !important;
        vertical-align: middle;
        border-left: 1.8px solid #222;
        border-right: 1.8px solid #222;
        border-collapse: collapse;
        margin: 0 4px;
        font-size: 9.8pt;
        font-style: normal;
    }
    table.bmatrix td {
        border: none !important;
        background: transparent !important;
        padding: 1px 6px;
        text-align: center;
    }
    span.frac {
        display: inline-block;
        vertical-align: middle;
        text-align: center;
        margin: 0 2px;
        line-height: 1.1;
    }
    span.frac > span.num {
        display: block;
        border-bottom: 1px solid #222;
        padding: 0 3px 1px 3px;
    }
    span.frac > span.den {
        display: block;
        padding: 1px 3px 0 3px;
    }
    blockquote {
        margin: 10px 0;
        padding: 8px 14px;
        border-left: 3.5px solid #8C1515;
        background-color: #faf8f8;
        color: #1f2328;
        border-radius: 0 4px 4px 0;
        page-break-inside: avoid;
    }
    figure.report-fig {
        margin: 12px auto;
        text-align: center;
        page-break-inside: avoid;
    }
    figure.report-fig img {
        max-width: 98%;
        height: auto;
        border: 1px solid #d0d7de;
        border-radius: 3px;
    }
    figure.report-fig figcaption {
        font-size: 8.9pt;
        color: #333333;
        margin-top: 5px;
        text-align: justify;
        padding: 0 8px;
    }
    .math-block {
        font-family: 'Cambria Math', 'Times New Roman', serif;
        font-size: 10.8pt;
        font-style: italic;
        text-align: center;
        margin: 9px 0;
        padding: 6px 10px;
        background-color: #fcfcfd;
        border-top: 1px solid #eaeef2;
        border-bottom: 1px solid #eaeef2;
        page-break-inside: avoid;
    }
    .math-inline {
        font-family: 'Cambria Math', 'Times New Roman', serif;
        font-style: italic;
    }
    a {
        color: #8C1515;
        text-decoration: none;
    }
    """

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<title>Báo Cáo Đồ Án IMP302 - Khôi Phục & Nhận Dạng Biển Số Xe</title>
<style>{css}</style>
</head>
<body>
{''.join(html_parts)}
</body>
</html>"""


def render_pdf(html_path: str, pdf_path: str):
    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)
    if os.path.exists(abs_pdf):
        os.remove(abs_pdf)

    browsers = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for b_path in browsers:
        if os.path.exists(b_path):
            cmd = [
                b_path,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={abs_pdf}",
                f"file:///{abs_html.replace(os.sep, '/')}",
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 10000:
                print(f"Successfully exported PDF via {os.path.basename(b_path)}: {pdf_path}")
                return True

    html_str = open(abs_html, encoding="utf-8").read()
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (36, 36, -36, -36)
    story = fitz.Story(html=html_str)
    writer = fitz.DocumentWriter(abs_pdf)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    print(f"Successfully exported PDF via PyMuPDF Story: {pdf_path}")
    return True


if __name__ == "__main__":
    html_out = "report/FINAL_REPORT.html"
    pdf_out1 = "report/FINAL_REPORT.pdf"
    pdf_out2 = "report/FINAL_REPORT_IMP302.pdf"

    html_doc = markdown_to_html_document("report/FINAL_REPORT.md")
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"Generated self-contained HTML preview: {html_out}")

    render_pdf(html_out, pdf_out1)
    shutil.copyfile(pdf_out1, pdf_out2)
    doc = fitz.open(pdf_out1)
    print(f"PDF Verification: {len(doc)} pages, size = {os.path.getsize(pdf_out1) / (1024*1024):.2f} MB")
    doc.close()
