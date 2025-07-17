import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path

def extract_text_from_pdf(pdf_path):
    final_text = []

    # Step 1️⃣ : Try extracting normal text using PyMuPDF
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # extract all text

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


# 🔥 Example Usage:
pdf_path = 'SOF.pdf'  # Replace with your PDF file name
text_output = extract_text_from_pdf(pdf_path)

# Save the extracted text to a file
with open('extracted_output.txt', 'w', encoding='utf-8') as f:
    f.write(text_output)

print("✅ Text extraction complete. Check 'extracted_output.txt'.")
