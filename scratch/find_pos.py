import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    text = f.read()

text_clean = re.sub(r'<svg.*?</svg>', '', text, flags=re.DOTALL)
text_clean = re.sub(r'<!--.*?-->', '', text_clean, flags=re.DOTALL)
text_clean = re.sub(r'<style.*?</style>', '', text_clean, flags=re.DOTALL)

lines = text_clean.splitlines()

for idx in [88, 102]:
    line = lines[idx]
    for m in re.finditer(r'\\[a-zA-Z]+', line):
        start = max(0, m.start() - 30)
        end = min(len(line), m.end() + 30)
        print(f"Index {idx+1} pos {m.start()}: {line[start:end]}")

