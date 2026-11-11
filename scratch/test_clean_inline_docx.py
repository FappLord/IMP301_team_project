import io
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

plt.rcParams['mathtext.fontset'] = 'cm'

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

test_text = r"Stage 2 localization is evaluated at $\text{IoU} \ge 0.50$ with MSE $\downarrow$ and PSNR $\uparrow$."
print("Inline clean result:", clean_tex_inline(test_text))

