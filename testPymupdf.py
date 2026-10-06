import fitz

pdf_path = "data/WMO-1368-2024_en.pdf"

doc = fitz.open(pdf_path)

print("Pages:", len(doc))

for page_number in [0, 1, 2, 10]:
    page = doc[page_number]

    print("\n" + "=" * 80)
    print(f"PAGE {page_number + 1}")
    print("=" * 80)

    print(page.get_text()[:2000])