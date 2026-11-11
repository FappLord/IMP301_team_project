import re

with open('report/FINAL_REPORT.html', 'r', encoding='utf-8') as f:
    html = f.read()

raws = re.findall(r'<span class=[\'\"]math-raw[\'\"]>.*?</span>', html)
print('Number of math-raw occurrences:', len(raws))
for r in raws:
    print('RAW:', r)

