import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    html = f.read()

rows = re.findall(r'<div class=[\'\"]math-row[\'\"]>.*?</div>\s*</div>', html, flags=re.DOTALL)
print('Total math rows found:', len(rows))
for i, r in enumerate(rows):
    num_m = re.search(r'<div class=[\'\"]math-num[\'\"]>(.*?)</div>', r)
    num = num_m.group(1) if num_m else f'unknown_{i}'
    has_svg = '<svg' in r
    has_raw = 'math-raw' in r
    print(f'Eq {num}: svg={has_svg}, raw={has_raw}')

