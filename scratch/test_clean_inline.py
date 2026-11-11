import sys, os
sys.path.insert(0, os.getcwd())
from build_pdf_report import clean_inline_math_html

test_cases = [
    r"\hat{\sigma}_\eta",
    r"\hat{f}_{\mathrm{WMF}}",
    r"\{LL_1, LH_1, HL_1, HH_1\}",
    r"\mathrm{Re}\{F^{-1}\{H_{\mathrm{NR}}(u,v)G(u,v)\}\}",
    r"D_{\pm k}(u,v)",
]

for tc in test_cases:
    print(f"{tc} -> {clean_inline_math_html(tc)}")
