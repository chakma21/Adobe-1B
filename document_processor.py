import fitz  # PyMuPDF library for PDF processing
import json    # For working with JSON data
import os      # For interacting with the operating system (e.g., file paths, directory creation)
from datetime import datetime # For generating timestamps
import re      # For regular expressions, useful for pattern matching in text

def extract_structured_text_from_pdf(pdf_path):
    """
    Extracts text blocks with basic styling information (font size, bold status)
    from a single PDF document.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list: A list of dictionaries, where each dictionary represents a page,
              and contains a list of lines with their text content and font properties.
              Returns an empty list if the PDF cannot be processed.
    """
    document_pages_structured_content = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_blocks_on_page = page.get_text("dict")["blocks"]
            
            page_content_lines = []
            for block in text_blocks_on_page:
                if block['type'] == 0:  # Check if it's a text block
                    for line in block['lines']:
                        line_text = ""
                        spans_info = []
                        for span in line['spans']:
                            line_text += span['text']
                            spans_info.append({
                                'text': span['text'],
                                'size': span['size'],
                                'font': span['font'],
                                'is_bold': bool(span['flags'] & 2) or ("bold" in span['font'].lower()) or ("heavy" in span['font'].lower())
# This now checks the flag OR if the font name contains "bold" or "heavy" (case-insensitive)
                            })
                        
                        is_potential_heading = False
                        avg_size = 0
                        is_bold_present = False

                        if spans_info:
                            avg_size = sum([s['size'] for s in spans_info]) / len(spans_info)
                            is_bold_present = any(s['is_bold'] for s in spans_info)
                            
                            # Heuristic for potential heading: adjust as needed
                            # The original heuristic was 'avg_size > 13 or is_bold_present'.
                            # Let's consider 12.0pt bold as a strong indicator for your specific PDFs.
                            # If the average font size is equal to or greater than 12.0 AND it's bold, OR if it's significantly larger
                            if (avg_size >= 12.0 and is_bold_present) or (avg_size > 13.0):
                                is_potential_heading = True

                        page_content_lines.append({
                            "text": line_text.strip(),
                            "is_potential_heading": is_potential_heading,
                            "font_size_avg": avg_size,
                            "is_bold_present": is_bold_present,
                            "original_spans": spans_info
                        })
            document_pages_structured_content.append({
                "page_number": page_num + 1,
                "lines": page_content_lines
            })
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []
    return document_pages_structured_content

def process_document_collection(input_json_data, documents_dir="documents/"):
    """
    Processes a collection of PDF documents and extracts structured text.
    """
    extracted_data = {
        "metadata": {
            "input_documents": [{"filename": doc["filename"], "title": doc["title"]} for doc in input_json_data["documents"]],
            "persona": input_json_data["persona"],
            "job_to_be_done": input_json_data["job_to_be_done"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_documents_structured_text": []
    }

    for doc_info in input_json_data["documents"]:
        filename = doc_info["filename"]
        title = doc_info["title"]
        pdf_path = os.path.join(documents_dir, filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: Document not found at {pdf_path}. Skipping.")
            continue 

        print(f"Extracting structured text from: {filename}")
        pages_structured_text = extract_structured_text_from_pdf(pdf_path)

        if pages_structured_text:
            extracted_data["extracted_documents_structured_text"].append({
                "filename": filename,
                "title": title,
                "pages": pages_structured_text
            })
    
    return extracted_data

def identify_document_sections(extracted_structured_data):
    """
    Identifies main sections within each document based on the structured text,
    leveraging font information.
    """
    extracted_sections = []

    for doc_data in extracted_structured_data["extracted_documents_structured_text"]:
        filename = doc_data["filename"]
        seen_section_titles_in_doc = set()

        for page_data in doc_data["pages"]:
            page_number = page_data["page_number"]
            for line_info in page_data["lines"]:
                text = line_info["text"].strip()
                
                if not text or len(text) < 5:
                    continue

                is_likely_section_title = False
                # Refined heuristic: Combine conditions to make it more robust
                # If it's flagged as a potential heading AND meets additional criteria
                if line_info["is_potential_heading"] and len(text) < 100: 
                    # Normalize text for checking against seen titles to avoid case sensitivity issues
                    normalized_text = text.lower()
                    if normalized_text not in seen_section_titles_in_doc:
                        # Stronger heuristic: all caps and short, or Title Case, or starting with common section words
                        if (text.isupper() and len(text.split()) < 10) or \
                           (text == text.title() and len(text.split()) < 15) or \
                           (re.match(r"^(Introduction|Chapter|Section|Overview|Key|Summary|Conclusion|Tips|Things to Do|Coastal|Culinary|Nightlife)", text, re.IGNORECASE)):
                            is_likely_section_title = True
                        # A general check for strong bold lines that might be headings
                        elif line_info["is_bold_present"] and line_info["font_size_avg"] >= 12.0:
                             is_likely_section_title = True

                if is_likely_section_title:
                    extracted_sections.append({
                        "document": filename,
                        "section_title": text,
                        "importance_rank": 0,
                        "page_number": page_number
                    })
                    seen_section_titles_in_doc.add(normalized_text) # Add normalized text to set

    return extracted_sections

# --- Main execution block ---
if __name__ == "__main__":
    input_json_filename = "input_payload.json"
    documents_directory = "documents"
    output_filename = "processed_document_output.json"
    
    # NEW INTERMEDIATE FILE
    intermediate_structured_text_output_filename = "intermediate_structured_text_data.json"

    # --- Step 1: Load Input Payload ---
    if not os.path.exists(input_json_filename):
        print(f"Error: Input JSON file '{input_json_filename}' not found.")
        print("Please ensure 'input_payload.json' is in the same directory as this script.")
        exit()

    try:
        with open(input_json_filename, 'r', encoding='utf-8') as f:
            input_payload = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_json_filename}': {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_json_filename}': {e}")
        exit()

    # --- Step 2: Ensure Documents Directory Exists ---
    os.makedirs(documents_directory, exist_ok=True)

    # --- Step 3: Extract Structured Text from PDFs ---
    print("Starting document text extraction...")
    extracted_structured_data = process_document_collection(input_payload, documents_dir=documents_directory)
    print("Document text extraction complete.")

    # --- DEBUGGING STEP: Save Intermediate Structured Text Data ---
    print(f"\nSaving intermediate structured text data to {intermediate_structured_text_output_filename}...")
    try:
        with open(intermediate_structured_text_output_filename, "w", encoding="utf-8") as f:
            json.dump(extracted_structured_data, f, indent=2)
        print(f"Intermediate structured data saved to {intermediate_structured_text_output_filename}")
    except Exception as e:
        print(f"Error saving intermediate structured data: {e}")


    # --- Step 4: Identify Sections from Structured Text ---
    print("\nIdentifying sections from extracted text...")
    identified_sections = identify_document_sections(extracted_structured_data)
    print("Section identification complete.")

    # --- Step 5: Prepare Final Output Structure ---
    final_output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in extracted_structured_data["metadata"]["input_documents"]],
            "persona": extracted_structured_data["metadata"]["persona"]["role"],
            "job_to_be_done": extracted_structured_data["metadata"]["job_to_be_done"]["task"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": identified_sections,
        "subsection_analysis": [] 
    }

    # --- Step 6: Save Final Output to JSON File ---
    print(f"\nSaving full processed output to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"Full processed output saved to {output_filename}")

    # Optional: Print a snippet of the identified sections for quick review
    print("\n--- Identified Sections (Snippet) ---")
    if final_output["extracted_sections"]:
        for i, section in enumerate(final_output["extracted_sections"][:5]):
            print(f"{i+1}. Document: {section['document']}, Section: '{section['section_title']}', Page: {section['page_number']}")
        if len(final_output["extracted_sections"]) > 5:
            print(f"... and {len(final_output['extracted_sections']) - 5} more sections.")
    else:
        print("No sections identified.")