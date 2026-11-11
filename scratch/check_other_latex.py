import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Strip svgs, comments, style
text_clean = re.sub(r'<svg.*?</svg>', '', text, flags=re.DOTALL)
text_clean = re.sub(r'<!--.*?-->', '', text_clean, flags=re.DOTALL)
text_clean = re.sub(r'<style.*?</style>', '', text_clean, flags=re.DOTALL)

for line_no, line in enumerate(text_clean.splitlines(), 1):
    m = re.findall(r'\\[a-zA-Z]+', line)
    if m:
        print(f"Line {line_no}: {m} -> {line.strip()[:100]}")

