import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Strip svgs
text_clean = re.sub(r'<svg.*?</svg>', '', text, flags=re.DOTALL)
text_clean = re.sub(r'<!--.*?-->', '', text_clean, flags=re.DOTALL)

matches = re.findall(r'<span class=[\'\"]math-inline[\'\"]>(.*?)</span>', text_clean)
found = set()
for m in matches:
    bs = re.findall(r'\\[a-zA-Z]+', m)
    if bs:
        print(f"INLINE MATH WITH LATEX: {m}")
        for b in bs:
            found.add(b)

print("All unhandled LaTeX commands in inline math:", found)

