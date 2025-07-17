import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path):
    final_text = []

    # Step 1️⃣ : Try extracting normal text using PyMuPDF
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        if text.strip():
            final_text.append(f"\n[Page {page_num+1} - Normal Text]\n{text}")
        else:
            final_text.append(f"\n[Page {page_num+1} - No Text Found. Using OCR...]")
    doc.close()

    # Step 2️⃣ : Use OCR for all pages (scanned PDFs, images)
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        ocr_text = pytesseract.image_to_string(image)
        if ocr_text.strip():
            final_text.append(f"\n[Page {i+1} - OCR Extracted Text]\n{ocr_text}")

    return "\n".join(final_text)


# 🔥 Example Usage: Handle Multiple PDFs
pdf_files = ['LA1.pdf', 'LA2.pdf', 'sample.pdf','SOF.pdf']  # Replace with your PDF filenames

output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)

for pdf_path in pdf_files:
    print(f"Processing {pdf_path}...")
    text_output = extract_text_from_pdf(pdf_path)
    
    # Generate a unique output filename
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(output_folder, f'{pdf_name}_extracted.txt')
    
    # Save the extracted text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text_output)

    print(f"✅ Extraction complete for {pdf_path}. Saved to {output_file}")

print("🎉 All PDFs processed.")
