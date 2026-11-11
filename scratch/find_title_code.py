with open('build_pdf_report.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if 'h1' in l.lower() or 'article-title' in l.lower() or 'doc-title' in l.lower():
        print(f"Line {i+1}: {l.strip()}")

