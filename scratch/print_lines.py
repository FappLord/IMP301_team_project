import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    text = f.read()

# Strip svgs, comments, style
text_clean = re.sub(r'<svg.*?</svg>', '', text, flags=re.DOTALL)
text_clean = re.sub(r'<!--.*?-->', '', text_clean, flags=re.DOTALL)
text_clean = re.sub(r'<style.*?</style>', '', text_clean, flags=re.DOTALL)

lines = text_clean.splitlines()
print("=== LINE 89 ===")
print(lines[88][:500])
print("=== LINE 103 ===")
print(lines[102][:500])

