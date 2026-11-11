import io
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

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

def render_math_to_png(tex: str, fontsize: float = 13) -> io.BytesIO:
    cleaned = clean_tex(tex)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${cleaned}$", fontsize=fontsize)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.04, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf

doc = docx.Document()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
buf = render_math_to_png(r"g(x,y) = \mathcal{T}_{\mathbf{H}}\{ (h * f)(x,y) \} + \eta(x,y)")
p.add_run().add_picture(buf)
doc.save('scratch/test_math.docx')
print('Successfully saved test_math.docx with rendered equation!')

