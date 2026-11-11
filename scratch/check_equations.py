import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    html = f.read()

displays = re.findall(r'<div class=[\'\"]math-display[\'\"]>(.*?)</div>\s*<div class=[\'\"]math-num[\'\"]>(\(\d+\))</div>', html, flags=re.DOTALL)
print('Total display equations found:', len(displays))
for content, num in displays:
    has_svg = '<svg' in content
    has_raw = 'math-raw' in content
    print(f'Eq {num}: svg={has_svg}, raw={has_raw}')

