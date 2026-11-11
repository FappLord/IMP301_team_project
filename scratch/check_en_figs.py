import re

# Read FINAL_REPORT.md to map sections and figure references
with open('report/FINAL_REPORT.md', 'r', encoding='utf-8') as f:
    en_text = f.read()

# Map en figures
en_figs = re.findall(r'!\[(.*?)\]\((.*?)\)', en_text)
print(f"Total figures in FINAL_REPORT.md: {len(en_figs)}")
for cap, path in en_figs:
    print(f"  {path} -> {cap}")

