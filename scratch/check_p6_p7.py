import pymupdf as fitz

doc = fitz.open('report/FINAL_REPORT.pdf')
print("=== PAGE 6 ===")
print(doc[5].get_text()[-400:])
print("=== PAGE 7 ===")
print(doc[6].get_text()[:400])

