import fitz  # PyMuPDF
import json
import os
from datetime import datetime
import re # Added for potential regex-based heading detection later, but not strictly used in this specific extract_structured_text_from_pdf version

def extract_structured_text_from_pdf(pdf_path):
    """
    Extracts text blocks with basic styling information (font size, bold)
    from a single PDF document.
    Returns a list of dictionaries, where each dictionary represents a block
    of text on a page, including its font properties.
    """
    document_pages_structured_content = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Use 'dict' output to get blocks, lines, and spans with styling
            # This provides detailed information about text layout and fonts
            text_blocks_on_page = page.get_text("dict")["blocks"]
            
            page_content_lines = []
            for block in text_blocks_on_page:
                if block['type'] == 0:  # Ensure it's a text block (type 1 is image)
                    for line in block['lines']:
                        line_text = ""
                        spans_info = []
                        # Iterate through spans to get text and font details
                        for span in line['spans']:
                            line_text += span['text']
                            spans_info.append({
                                'text': span['text'],
                                'size': span['size'],
                                'font': span['font'],
                                'is_bold': bool(span['flags'] & 2) # Flag 2 indicates bold
                            })
                        
                        # Basic heuristic for potential heading:
                        # A line is considered a potential heading if it's not empty,
                        # has an average font size above a certain threshold (e.g., 13pt),
                        # or contains bold text. These thresholds might need tuning.
                        is_potential_heading = False
                        avg_size = 0
                        is_bold_present = False

                        if spans_info:
                            avg_size = sum([s['size'] for s in spans_info]) / len(spans_info)
                            is_bold_present = any(s['is_bold'] for s in spans_info)
                            
                            # Adjust threshold based on typical body text size in your PDFs
                            if avg_size > 13 or is_bold_present:
                                is_potential_heading = True

                        page_content_lines.append({
                            "text": line_text.strip(),
                            "is_potential_heading": is_potential_heading,
                            "font_size_avg": avg_size,
                            "is_bold_present": is_bold_present,
                            "original_spans": spans_info # Keep original span info for deeper analysis if needed
                        })
            document_pages_structured_content.append({
                "page_number": page_num + 1,
                "lines": page_content_lines # Renamed from 'blocks' to 'lines' to reflect granularity
            })
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []
    return document_pages_structured_content

def process_document_collection(input_json_data, documents_dir="documents/"):
    """
    Processes a collection of PDF documents based on the input JSON
    and extracts structured text with styling information.
    """
    extracted_data = {
        "metadata": {
            "input_documents": [],
            "persona": input_json_data["persona"],
            "job_to_be_done": input_json_data["job_to_be_done"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_documents_structured_text": [] # Changed key name to reflect structured output
    }

    # Populate input documents metadata
    for doc_info in input_json_data["documents"]:
        extracted_data["metadata"]["input_documents"].append({
            "filename": doc_info["filename"],
            "title": doc_info["title"]
        })

    # Process each document
    for doc_info in input_json_data["documents"]:
        filename = doc_info["filename"]
        title = doc_info["title"]
        pdf_path = os.path.join(documents_dir, filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: Document not found at {pdf_path}. Skipping.")
            continue # Skip to the next document if file not found

        print(f"Extracting structured text from: {filename}")
        # Call the new structured extraction function
        pages_structured_text = extract_structured_text_from_pdf(pdf_path)

        # Only add document to extracted_documents_structured_text if extraction was successful
        if pages_structured_text:
            extracted_data["extracted_documents_structured_text"].append({
                "filename": filename,
                "title": title,
                "pages": pages_structured_text # Now contains structured lines with font info
            })
    
    return extracted_data

if __name__ == "__main__":
    input_json_filename = "input_payload.json"
    documents_directory = "documents" # Make sure this directory exists and contains your PDFs

    # Check if the input JSON file exists
    if not os.path.exists(input_json_filename):
        print(f"Error: Input JSON file '{input_json_filename}' not found.")
        print("Please create 'input_payload.json' in the same directory as the script.")
        exit()

    # Load the input JSON data from the file
    try:
        with open(input_json_filename, 'r', encoding='utf-8') as f:
            input_payload = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{input_json_filename}': {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_json_filename}': {e}")
        exit()

    # Ensure the documents directory exists
    os.makedirs(documents_directory, exist_ok=True)

    # Process the document collection based on the loaded input payload
    extracted_output = process_document_collection(input_payload, documents_dir=documents_directory)

    print("\n--- Extracted Structured Document Text ---")
    # For a large output, you might want to print only a snippet or save to file
    # print(json.dumps(extracted_output, indent=2)) 

    # Save the output to a JSON file
    output_filename = "extracted_structured_document_data.json" # New output filename
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(extracted_output, f, indent=2)
    print(f"\nExtracted structured data saved to {output_filename}")