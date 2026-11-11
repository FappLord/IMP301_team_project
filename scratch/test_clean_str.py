import re

def clean_inline_math_test(s: str) -> str:
    s = s.strip()
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    # clean backslashes
    s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)
    return s

test = r"\mathrm{Re}\{\mathcal{F}^{-1}\{H_{\mathrm{NR}}(u,v)G(u,v)\}\}"
print("Before:", test)
print("After:", clean_inline_math_test(test))

