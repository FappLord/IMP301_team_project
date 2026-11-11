with open('build_pdf_report.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '<style>' in l or '.math-display-row' in l:
        print(f"Line {i+1}: {l.strip()}")

