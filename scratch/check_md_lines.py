with open('report/FINAL_REPORT.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'H_{' in l or 'HNR' in l or 'Butterworth' in l:
        print(f"Line {i+1}: {l.strip()[:120]}")

