import pdfplumber
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
pdf_path = root_dir / "PART_2" / "pdf" / "Gaza_genocide.pdf"
txt_path = root_dir / "PART_2" / "gaza_genocide.txt"

with pdfplumber.open(pdf_path) as pdf:
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        for page in pdf.pages:
            texts = page.extract_text()
            if texts:
                txt_file.write(texts)
                txt_file.write("\n")
print(f"Text extracted and saved to {txt_path}")