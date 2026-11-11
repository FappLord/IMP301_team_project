import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    html = f.read()

for m in re.finditer(r'class=[\'\"][^\'\"]*math[^\'\"]*[\'\"]', html):
    print(m.group(0))

