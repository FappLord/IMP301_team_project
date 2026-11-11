import os
import re
import html
import base64
import shutil
import subprocess
import pymupdf as fitz
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"


def clean_tex_for_mathtext(s: str) -> str:
    """Prepare LaTeX equation string for Matplotlib mathtext engine."""
    s = s.strip()
    s = s.replace(r"\underbrace", "")
    s = re.sub(r"\\le\b", r"\\leq", s)
    s = re.sub(r"\\ge\b", r"\\geq", s)
    s = re.sub(r"_\{([^}]+)\\text\{ times\}\}", r"_{\1}", s)
    s = re.sub(r"_\{([^}]+)\\mathrm\{ times\}\}", r"_{\1}", s)
    s = s.replace(r"\mathbb{I}", r"\mathbf{I}")
    s = s.replace(r"\mathbb{R}", r"\mathrm{R}")
    s = s.replace(r"\mathcal{R}e", r"\mathrm{Re}")
    s = s.replace(r"\mathcal{F}", r"\mathcal{F}")
    s = s.replace(r"\mathcal{T}", r"\mathcal{T}")
    s = s.replace(r"\text", r"\mathrm")
    s = s.replace(r"\mathbf", r"\mathbf")
    s = s.replace(r"\qquad", r"\quad\quad")
    s = s.replace(r"\%", "%")
    for macro in [r"\left", r"\right", r"\Big", r"\big", r"\!", r"\,", r"\;"]:
        s = s.replace(macro, "")
    return s


def render_equation_to_svg(tex: str, fontsize: float = 11.5) -> str:
    """Render LaTeX equation to an inline SVG string using Matplotlib's TeX math engine."""
    clean_tex = clean_tex_for_mathtext(tex)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${clean_tex}$", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    svg_code = buf.getvalue().decode("utf-8")
    svg_code = re.sub(r"<\?xml.*?\?>", "", svg_code)
    svg_code = re.sub(r"<!DOCTYPE.*?>", "", svg_code)
    return svg_code.strip()


def format_display_equation(raw_eq: str) -> str:
    """Format display equation block ($$...$$), extracting equation numbers if present."""
    raw = raw_eq.strip()
    eq_num = None
    m_eq = re.search(r"\\qquad\s*\((\d+)\)\s*$", raw)
    if m_eq:
        eq_num = m_eq.group(1)
        raw = raw[: m_eq.start()].strip()

    if r"\begin{bmatrix}" in raw:
        def _repl_bm(m):
            body = m.group(1).strip()
            rows = [r.strip() for r in body.split(r"\\") if r.strip()]
            row_html = []
            for r in rows:
                cols = [c.strip() for c in r.split("&")]
                tds = "".join(f"<td>{render_equation_to_svg(c, fontsize=10.5)}</td>" for c in cols)
                row_html.append(f"<tr>{tds}</tr>")
            return f"<table class='bmatrix'>{''.join(row_html)}</table>"

        tokens = re.split(r"(\\begin\{bmatrix\}.*?\\end\{bmatrix\})", raw, flags=re.DOTALL)
        content_parts = []
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok.startswith(r"\begin{bmatrix}"):
                content_parts.append(_repl_bm(re.match(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", tok, flags=re.DOTALL)))
            else:
                try:
                    content_parts.append(render_equation_to_svg(tok))
                except Exception:
                    content_parts.append(f"<span class='math-inline'>{html.escape(tok)}</span>")
        content = " ".join(content_parts)
    else:
        try:
            content = render_equation_to_svg(raw)
        except Exception:
            content = f"<span class='math-raw'>${html.escape(raw)}$</span>"

    if eq_num:
        return (
            f"<div class='math-display-row'>"
            f"<div class='math-svg-center'>{content}</div>"
            f"<div class='math-num'>({eq_num})</div>"
            f"</div>"
        )
    return f"<div class='math-display-row'><div class='math-svg-center'>{content}</div></div>"


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
    """Safely replace \\macro{arg1}{arg2} supporting arbitrary nested braces."""
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
    """Safely replace \\macro{arg1} supporting arbitrary nested braces."""
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


def clean_inline_math_html(s: str) -> str:
    """Format short inline math expressions into clean HTML entities with proper math styling."""
    s = s.strip()

    # Macro replacements with nested brace support
    for _ in range(3):
        s = replace_two_arg_macro(
            s,
            r"\frac",
            lambda a, b: f"<span class='frac'><span class='num'>{a}</span><span class='den'>{b}</span></span>",
        )
        s = replace_one_arg_macro(
            s,
            r"\sqrt",
            lambda a: f"&radic;<span style='border-top:1px solid #222;padding-top:1px;'>{a}</span>",
        )
        s = replace_one_arg_macro(
            s,
            r"\text",
            lambda a: f"<span style='font-style:normal;font-family:Times New Roman,serif;'>{a}</span>",
        )
        s = replace_one_arg_macro(
            s,
            r"\mathrm",
            lambda a: f"<span style='font-style:normal;font-family:Times New Roman,serif;'>{a}</span>",
        )
        s = replace_one_arg_macro(s, r"\mathbf", lambda a: f"<b>{a}</b>")
        s = replace_one_arg_macro(s, r"\mathcal", lambda a: f"<i><b>{a}</b></i>")
        s = replace_one_arg_macro(s, r"\mathbb", lambda a: f"<b>{a}</b>")
        s = replace_one_arg_macro(s, r"\hat", lambda a: f"{a}&#770;")
        s = replace_one_arg_macro(s, r"\bar", lambda a: f"{a}&#772;")
        s = replace_one_arg_macro(s, r"\tilde", lambda a: f"{a}&#771;")

    reps = {
        r"\in": "&isin;",
        r"\notin": "&notin;",
        r"\cup": "&cup;",
        r"\cap": "&cap;",
        r"\subset": "&sub;",
        r"\le": "&le;",
        r"\leq": "&le;",
        r"\ge": "&ge;",
        r"\geq": "&ge;",
        r"\uparrow": "&uarr;",
        r"\downarrow": "&darr;",
        r"\forall": "&forall;",
        r"\exists": "&exist;",
        r"\ll": "&Lt;",
        r"\gg": "&Gt;",
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
        r"\to": "&rarr;",
        r"\rightarrow": "&rarr;",
        r"\leftarrow": "&larr;",
        r"\implies": "&rArr;",
        r"\infty": "&infin;",
        r"\sum": "&sum;",
        r"\prod": "&prod;",
        r"\int": "&int;",
        r"\Delta": "&Delta;",
        r"\Omega": "&Omega;",
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
        r"\tau": "&tau;",
        r"\rho": "&rho;",
        r"\phi": "&phi;",
        r"\psi": "&psi;",
        r"\partial": "&part;",
        r"\mathbb{R}": "&#8477;",
        r"\mathbb{I}": "<b>I</b>",
        r"\dots": "&hellip;",
        r"\cdots": "&ctdot;",
        r"\|": "&#8214;",
        r"\ln": "<span style='font-style:normal;'>ln</span>",
        r"\log": "<span style='font-style:normal;'>log</span>",
        r"\sin": "<span style='font-style:normal;'>sin</span>",
        r"\cos": "<span style='font-style:normal;'>cos</span>",
        r"\max": "<span style='font-style:normal;'>max</span>",
        r"\min": "<span style='font-style:normal;'>min</span>",
        r"\arg": "<span style='font-style:normal;'>arg</span>",
    }
    for k, v in reps.items():
        s = s.replace(k, v)

    for _ in range(3):
        s = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_(&[a-zA-Z]+;)", r"<sub>\1</sub>", s)
    s = re.sub(r"\^(&[a-zA-Z]+;)", r"<sup>\1</sup>", s)
    s = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([a-zA-Z0-9\*])", r"<sup>\1</sup>", s)

    for macro in [r"\Big", r"\big", r"\left", r"\right", r"\!", r"\,", r"\;"]:
        s = s.replace(macro, "")
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace(r"\%", "%")
    return f"<span class='math-inline'>{s}</span>"


def format_inline_markdown(text: str) -> str:
    """Format links, bold, italic, code spans, and inline math."""
    code_map = {}
    def _save_code(m):
        k = f"@@CODE{len(code_map)}@@"
        code_map[k] = f"<code>{html.escape(m.group(1))}</code>"
        return k
    text = re.sub(r"`([^`]+)`", _save_code, text)

    math_map = {}
    def _save_math(m):
        k = f"@@MATH{len(math_map)}@@"
        math_map[k] = clean_inline_math_html(m.group(1))
        return k
    text = re.sub(r"\$([^$\n]+)\$", _save_math, text)

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    for k, v in math_map.items():
        text = text.replace(k, v)
    for k, v in code_map.items():
        text = text.replace(k, v)
    return text


def markdown_to_html_document(md_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(md_path))
    lines = open(md_path, encoding="utf-8").read().splitlines()

    html_parts = []
    i = 0
    n = len(lines)
    fig_counter = 0

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Raw HTML block e.g. author grid
        if stripped.startswith("<div"):
            block = [line]
            i += 1
            while i < n and not lines[i].strip().startswith("</div>"):
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i])
                i += 1
            html_parts.append("\n".join(block))
            continue

        # Algorithm 1 block
        if stripped == "---" and i + 1 < n:
            peek = i + 1
            while peek < n and not lines[peek].strip():
                peek += 1
            if peek < n and lines[peek].strip().startswith("**Algorithm"):
                alg_title = format_inline_markdown(lines[peek].strip())
                i = peek + 1
                alg_body = []
                while i < n and lines[i].strip() != "---":
                    s_alg = lines[i].strip()
                    if s_alg:
                        if s_alg.startswith("- "):
                            alg_body.append(f"<div class='alg-req'>{format_inline_markdown(s_alg[2:])}</div>")
                        else:
                            alg_body.append(f"<div class='alg-line'>{format_inline_markdown(s_alg)}</div>")
                    i += 1
                if i < n and lines[i].strip() == "---":
                    i += 1
                html_parts.append(
                    f"<div class='alg-box'>"
                    f"<div class='alg-header'>{alg_title}</div>"
                    f"<div class='alg-body'>{''.join(alg_body)}</div>"
                    f"</div>"
                )
                continue

        if stripped == "---":
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
            escaped = html.escape("\n".join(code_lines))
            html_parts.append(f"<pre class='code-block'><code class='lang-{lang}'>{escaped}</code></pre>")
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
            eq_raw = " ".join(math_lines)
            html_parts.append(format_display_equation(eq_raw))
            continue

        # Markdown Table
        if stripped.startswith("|") and stripped.endswith("|"):
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            caption_html = ""
            peek = i
            while peek < n and not lines[peek].strip():
                peek += 1
            if peek < n and re.match(r"^\*Table\s+\d+:", lines[peek].strip()):
                cap_raw = lines[peek].strip().strip("*")
                caption_html = f"<div class='table-caption'>{format_inline_markdown(cap_raw)}</div>"
                i = peek + 1

            rows_html = []
            is_header = True
            for tline in tbl_lines:
                s_row = tline.strip()
                if s_row.startswith("|"):
                    s_row = s_row[1:]
                if s_row.endswith("|"):
                    s_row = s_row[:-1]
                cells, buf, in_m, in_c = [], "", False, False
                for ch in s_row:
                    if ch == "`" and not in_m:
                        in_c = not in_c
                    elif ch == "$" and not in_c:
                        in_m = not in_m
                    if ch == "|" and not in_m and not in_c:
                        cells.append(buf.strip())
                        buf = ""
                    else:
                        buf += ch
                cells.append(buf.strip())

                if all(re.match(r"^[:\-]+$", c.replace(" ", "")) for c in cells if c):
                    continue
                tag = "th" if is_header else "td"
                row_cells = "".join(f"<{tag}>{format_inline_markdown(c)}</{tag}>" for c in cells)
                rows_html.append(f"<tr>{row_cells}</tr>")
                is_header = False

            html_parts.append(
                f"<div class='table-wrap'>{caption_html}<table class='data-table'>{''.join(rows_html)}</table></div>"
            )
            continue

        # Headings #
        if stripped.startswith("#"):
            lvl = min(len(stripped) - len(stripped.lstrip("#")), 4)
            h_title = format_inline_markdown(stripped.lstrip("#").strip())
            if "BÁO CÁO ĐỒ ÁN MÔN HỌC" in h_title or "COURSE PROJECT REPORT" in h_title:
                html_parts.append(
                    f"<div class='report-header-box'>"
                    f"<div class='report-top-title'>{h_title}</div>"
                    f"<hr class='report-header-divider'/>"
                    f"</div>"
                )
            elif "ĐỀ TÀI:" in h_title or "TOPIC:" in h_title:
                html_parts.append(f"<div class='report-topic-title'>{h_title}</div>")
            else:
                html_parts.append(f"<h{lvl}>{h_title}</h{lvl}>")
            i += 1
            continue

        # Subtitle italic line
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**") and ("Multi-Degradation" in stripped or "BÁO CÁO" in stripped or "Subtitle" in stripped):
            sub_t = format_inline_markdown(stripped.strip("*"))
            html_parts.append(f"<div class='report-sub-title'>{sub_t}</div>")
            i += 1
            continue

        # Image ![alt](path)
        if stripped.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                caption, rel_path = m.group(1), m.group(2)
                img_abs = os.path.normpath(os.path.join(base_dir, rel_path))
                if os.path.exists(img_abs):
                    fig_counter += 1
                    with open(img_abs, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    fig_prefix = "Hình" if "_VI" in os.path.basename(md_path) else "Figure"
                    html_parts.append(
                        f"<figure class='report-fig'>"
                        f"<img src='data:image/png;base64,{b64}' alt='{html.escape(caption)}'/>"
                        f"<figcaption><b>{fig_prefix} {fig_counter}:</b> {format_inline_markdown(caption)}.</figcaption>"
                        f"</figure>"
                    )
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            bq = []
            while i < n and lines[i].strip().startswith(">"):
                bq.append(format_inline_markdown(lines[i].strip().lstrip(">").strip()))
                i += 1
            html_parts.append(f"<blockquote>{'<br/>'.join(bq)}</blockquote>")
            continue

        # Unordered list - or *
        if stripped.startswith("- ") or stripped.startswith("* "):
            items = []
            while i < n and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                items.append(f"<li>{format_inline_markdown(lines[i].strip()[2:].strip())}</li>")
                i += 1
            html_parts.append(f"<ul>{''.join(items)}</ul>")
            continue

        # Ordered list
        m_num = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m_num:
            items = []
            while i < n:
                m_cur = re.match(r"^(\d+)\.\s+(.*)", lines[i].strip())
                if not m_cur:
                    break
                num_v, item_t = m_cur.group(1), m_cur.group(2)
                i += 1
                items.append(f"<li value='{num_v}'>{format_inline_markdown(item_t)}</li>")
            html_parts.append(f"<ol>{''.join(items)}</ol>")
            continue

        # Paragraph
        if re.match(r"^\[\d+\]\s+", stripped):
            html_parts.append(f"<p class='ref-item'>{format_inline_markdown(stripped)}</p>")
        else:
            html_parts.append(f"<p>{format_inline_markdown(stripped)}</p>")
        i += 1

    css = """
    @page {
        size: A4 portrait;
        margin: 14mm 14mm 15mm 14mm;
    }
    body {
        font-family: 'Computer Modern Serif', 'Latin Modern Roman', 'Times New Roman', Georgia, serif;
        font-size: 9.8pt;
        line-height: 1.44;
        color: #111111;
        max-width: 900px;
        margin: 0 auto;
        padding: 0;
        background: #ffffff;
    }
    .report-header-box {
        text-align: center;
        margin-top: 6px;
        margin-bottom: 2px;
    }
    .report-top-title {
        font-family: 'Times New Roman', Georgia, serif;
        font-size: 16.5pt;
        font-weight: 800;
        color: #123e67;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 5px;
        line-height: 1.3;
    }
    .report-header-divider {
        border: none;
        border-top: 2.5px solid #123e67;
        margin: 4px auto 8px auto;
        width: 100%;
    }
    .report-topic-title {
        font-family: 'Times New Roman', Georgia, serif;
        font-size: 12.8pt;
        font-weight: 800;
        color: #123e67;
        text-transform: uppercase;
        line-height: 1.38;
        text-align: center;
        margin-bottom: 10px;
    }
    .report-sub-title {
        font-family: 'Times New Roman', Georgia, serif;
        font-size: 10.5pt;
        font-style: italic;
        color: #333333;
        text-align: center;
        margin-bottom: 12px;
        line-height: 1.35;
    }
    h1 {
        font-family: 'Times New Roman', Georgia, serif;
        font-size: 16.5pt;
        font-weight: 700;
        color: #123e67;
        padding: 8px 6px;
        margin-top: 4px;
        margin-bottom: 10px;
        text-align: center;
        line-height: 1.28;
    }
    .author-grid {
        display: flex;
        justify-content: space-around;
        align-items: center;
        text-align: center;
        margin: 4px 0 18px 0;
        padding-bottom: 6px;
        font-size: 10.4pt;
        line-height: 1.35;
    }
    .author-item {
        flex: 1;
        padding: 0 4px;
    }
    h2 {
        font-family: 'Computer Modern Serif', 'Times New Roman', Georgia, serif;
        font-size: 12.5pt;
        font-weight: 700;
        color: #000000;
        border-bottom: 1px solid #e1e4e8;
        padding-bottom: 2px;
        margin-top: 18px;
        margin-bottom: 7px;
        page-break-after: avoid;
    }
    h3 {
        font-family: 'Computer Modern Serif', 'Times New Roman', Georgia, serif;
        font-size: 10.8pt;
        font-weight: 700;
        color: #000000;
        margin-top: 13px;
        margin-bottom: 5px;
        page-break-after: avoid;
    }
    h4 {
        font-family: 'Computer Modern Serif', 'Times New Roman', Georgia, serif;
        font-size: 10pt;
        font-weight: 700;
        font-style: italic;
        color: #222222;
        margin-top: 10px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }
    p, li {
        text-align: justify;
        margin-top: 0;
        margin-bottom: 6px;
    }
    p.ref-item {
        font-size: 9pt;
        line-height: 1.36;
        padding-left: 20px;
        text-indent: -20px;
        margin-bottom: 5px;
    }
    ul, ol {
        margin-top: 3px;
        margin-bottom: 7px;
        padding-left: 22px;
    }
    code {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8.6pt;
        background-color: #f4f5f7;
        color: #111111;
        padding: 1px 3.5px;
        border-radius: 2px;
    }
    pre.code-block {
        background-color: #f8f9fa;
        border: 1px solid #d0d7de;
        border-radius: 3px;
        padding: 8px 10px;
        font-size: 8.4pt;
        line-height: 1.34;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    pre.code-block code {
        background: none;
        padding: 0;
    }
    .alg-box {
        border-top: 1.8px solid #000000;
        border-bottom: 1.8px solid #000000;
        margin: 11px 0 14px 0;
        padding: 4px 6px;
        font-size: 9pt;
        line-height: 1.36;
        page-break-inside: avoid;
        background: #fafbfc;
    }
    .alg-header {
        border-bottom: 1px solid #000000;
        padding-bottom: 4px;
        margin-bottom: 5px;
        text-align: left;
    }
    .alg-req {
        margin-bottom: 3px;
        text-align: left;
    }
    .alg-line {
        text-align: left;
        margin-bottom: 2px;
    }
    .table-wrap {
        margin: 12px 0 15px 0;
        page-break-inside: avoid;
    }
    .table-caption {
        font-size: 9.2pt;
        color: #111111;
        margin-bottom: 4px;
        text-align: left;
    }
    table.data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8.6pt;
        line-height: 1.32;
        border-top: 1.8px solid #000000;
        border-bottom: 1.8px solid #000000;
    }
    table.data-table th, table.data-table td {
        border-bottom: 0.6px solid #d0d5dd;
        border-left: none;
        border-right: none;
        padding: 5px 6.5px;
        vertical-align: top;
        text-align: left;
    }
    table.data-table th {
        color: #000000;
        font-weight: 700;
        border-bottom: 1.2px solid #000000;
        background-color: #f6f8fa;
    }
    table.data-table tr:last-child td {
        border-bottom: none;
    }
    table.data-table tr:nth-child(even) td {
        background-color: #fafbfc;
    }
    table.bmatrix {
        display: inline-table;
        width: auto !important;
        vertical-align: middle;
        border-left: 1.5px solid #111;
        border-right: 1.5px solid #111;
        border-collapse: collapse;
        margin: 0 4px;
        font-size: 9.5pt;
        font-style: normal;
    }
    table.bmatrix td {
        border: none !important;
        background: transparent !important;
        padding: 1px 5px;
        text-align: center;
    }
    span.frac {
        display: inline-block;
        vertical-align: middle;
        text-align: center;
        margin: 0 2px;
        line-height: 1.05;
    }
    span.frac > span.num {
        display: block;
        border-bottom: 0.9px solid #111;
        padding: 0 2px 1px 2px;
    }
    span.frac > span.den {
        display: block;
        padding: 1px 2px 0 2px;
    }
    figure.report-fig {
        margin: 13px auto 15px auto;
        text-align: center;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        break-inside: avoid-page !important;
    }
    figure.report-fig img {
        max-width: 98%;
        height: auto;
        border: 0.8px solid #d0d7de;
        border-radius: 2px;
    }
    figure.report-fig figcaption {
        font-size: 9pt;
        color: #111111;
        margin-top: 5px;
        text-align: justify;
        padding: 0 10px;
    }
    .math-display-row {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px 0;
        padding: 4px 0;
        page-break-inside: avoid !important;
        break-inside: avoid !important;
        break-inside: avoid-page !important;
        position: relative;
    }
    .math-svg-center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .math-svg-center svg {
        vertical-align: middle;
        max-width: 90%;
        height: auto;
    }
    .math-num {
        position: absolute;
        right: 6px;
        font-size: 9.5pt;
        color: #111111;
        font-style: normal;
    }
    .math-inline {
        font-family: 'Cambria Math', 'Times New Roman', serif;
        font-style: italic;
    }
    a {
        color: #000000;
        text-decoration: none;
    }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Multi-Degradation Image Restoration, Perspective Homography Rectification, and Hybrid Transform-Topological Pattern Recognition for Two-Line License Plates</title>
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
                doc = fitz.open(abs_pdf)
                for idx, page in enumerate(doc):
                    rect = page.rect
                    page.insert_text(
                        fitz.Point(rect.width / 2 - 4, rect.height - 18),
                        str(idx + 1),
                        fontsize=9.5,
                        fontname="tiro",
                        color=(0, 0, 0),
                    )
                tmp_pdf = abs_pdf + ".tmp.pdf"
                doc.save(tmp_pdf)
                doc.close()
                os.replace(tmp_pdf, abs_pdf)
                print(f"Successfully exported PDF via {os.path.basename(b_path)}: {pdf_path}")
                return True
    return False


if __name__ == "__main__":
    targets = [
        ("report/FINAL_REPORT.md", "report/FINAL_REPORT.html", "report/FINAL_REPORT.pdf", "report/FINAL_REPORT_IMP302.pdf"),
        ("report/FINAL_REPORT_VI.md", "report/FINAL_REPORT_VI.html", "report/FINAL_REPORT_VI.pdf", "report/FINAL_REPORT_VI_IMP302.pdf"),
    ]

    for md_in, html_out, pdf_out1, pdf_out2 in targets:
        if not os.path.exists(md_in):
            continue
        print(f"\n==========================================")
        print(f"Building: {md_in} -> {pdf_out1}")
        print(f"==========================================")
        html_doc = markdown_to_html_document(md_in)
        with open(html_out, "w", encoding="utf-8") as f:
            f.write(html_doc)
        print(f"Generated self-contained HTML: {html_out}")

        success = render_pdf(html_out, pdf_out1)
        if success and os.path.exists(pdf_out1):
            shutil.copyfile(pdf_out1, pdf_out2)
            doc = fitz.open(pdf_out1)
            print(f"PDF Verification ({pdf_out1}): {len(doc)} pages, size = {os.path.getsize(pdf_out1) / (1024*1024):.2f} MB")
            doc.close()
