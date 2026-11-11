import io
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

plt.rcParams['mathtext.fontset'] = 'cm'

def clean_tex(s):
    s = s.strip()
    s = s.replace(r'\underbrace', '')
    s = re.sub(r'\\le\b', r'\\leq', s)
    s = re.sub(r'\\ge\b', r'\\geq', s)
    s = re.sub(r'_\{([^}]+)\\text\{ times\}\}', r'_{\1}', s)
    s = re.sub(r'_\{([^}]+)\\mathrm\{ times\}\}', r'_{\1}', s)
    s = s.replace(r'\mathbb{I}', r'\mathbf{I}')
    s = s.replace(r'\mathbb{R}', r'\mathrm{R}')
    s = s.replace(r'\text', r'\mathrm')
    s = s.replace(r'\%', '%')
    for macro in [r'\left', r'\right', r'\Big', r'\big', r'\!', r'\,', r'\;']:
        s = s.replace(macro, '')
    return s

def render_math_to_png(tex: str, fontsize: float = 12.5) -> io.BytesIO:
    cleaned = clean_tex(tex)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${cleaned}$", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.03, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

doc = docx.Document()
tbl = doc.add_table(rows=1, cols=2)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
# Remove borders
for cell in tbl.rows[0].cells:
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = docx.oxml.OxmlElement('w:tcBorders')
    for b in ['top', 'left', 'bottom', 'right']:
        edge = docx.oxml.OxmlElement(f'w:{b}')
        edge.set(docx.oxml.ns.qn('w:val'), 'none')
        tcBorders.append(edge)
    tcPr.append(tcBorders)

c0 = tbl.cell(0, 0)
c1 = tbl.cell(0, 1)
c0.width = Inches(5.8)
c1.width = Inches(0.7)

p0 = c0.paragraphs[0]
p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
buf = render_math_to_png(r"g(x,y) = \mathcal{T}_{\mathbf{H}}\{ (h * f)(x,y) \} + \eta(x,y)")
p0.add_run().add_picture(buf)

p1 = c1.paragraphs[0]
p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
p1.add_run("(1)")

doc.save('scratch/test_math_table.docx')
print('Successfully saved test_math_table.docx!')

