import io
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    s = s.replace(r'\mathcal{R}e', r'\mathrm{Re}')
    s = s.replace(r'\text', r'\mathrm')
    s = s.replace(r'\%', '%')
    for macro in [r'\left', r'\right', r'\Big', r'\big', r'\!', r'\,', r'\;']:
        s = s.replace(macro, '')
    return s

with open('report/FINAL_REPORT.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

eq_lines = [l.strip() for l in lines if l.strip().startswith('$$')]
print(f"Found {len(eq_lines)} display equations in FINAL_REPORT.md:")

for idx, el in enumerate(eq_lines, 1):
    raw = el.strip('$').strip()
    m_eq = re.search(r'\\qquad\s*\((\d+)\)\s*$', raw)
    eq_num = m_eq.group(1) if m_eq else str(idx)
    if m_eq:
        raw = raw[:m_eq.start()].strip()
    
    if r'\begin{bmatrix}' in raw:
        print(f"Eq ({eq_num}): Contains bmatrix (handled with table/clean notation)")
    else:
        try:
            c = clean_tex(raw)
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.text(0, 0, f"${c}$", fontsize=12)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', pad_inches=0.03, transparent=True)
            plt.close(fig)
            print(f"Eq ({eq_num}): Successfully rendered PNG ({len(buf.getvalue())} bytes)")
        except Exception as e:
            print(f"Eq ({eq_num}): FAILED with {e}")

