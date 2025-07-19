import fitz  # PyMuPDF library for PDF processing
import json    # For working with JSON data
import os      # For interacting with the operating system (e.g., file paths, directory creation)
import re      # For regular expressions, crucial for heading patterns

def extract_structured_text_from_pdf_single(pdf_path):
    """
    Extracts text blocks with styling information from a single PDF document.
    Modified to improve bold detection from font names.

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
                            # Improved is_bold detection: check flags OR font name
                            is_span_bold = bool(span['flags'] & 2) or \
                                           ("bold" in span['font'].lower()) or \
                                           ("heavy" in span['font'].lower()) or \
                                           ("black" in span['font'].lower()) # Added 'black' for some font families

                            spans_info.append({
                                'text': span['text'],
                                'size': span['size'],
                                'font': span['font'],
                                'is_bold': is_span_bold
                            })
                        
                        is_potential_heading = False
                        avg_size = 0
                        is_bold_present = False

                        if spans_info:
                            avg_size = sum([s['size'] for s in spans_info]) / len(spans_info)
                            is_bold_present = any(s['is_bold'] for s in spans_info)
                            
                            # Heuristic for potential heading:
                            # A line is a potential heading if it's bold AND at least a certain size (e.g., 12.0pt)
                            # OR if it's significantly larger than typical body text (e.g., > 13.0pt)
                            if (is_bold_present and avg_size >= 12.0) or (avg_size > 13.0):
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

def extract_document_outline(pdf_path):
    """
    Extracts document title and a structured outline (headings with levels) from a single PDF.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        dict: A dictionary containing the document title and the extracted outline.
              Example: {"title": "Document Title", "outline": [{"level": "H1", "text": "Section", "page": 1}]}
    """
    structured_data = extract_structured_text_from_pdf_single(pdf_path)
    if not structured_data:
        return {"title": os.path.basename(pdf_path), "outline": []}

    document_outline = []
    document_title = os.path.basename(pdf_path).replace(".pdf", "").replace("_", " ").title() # Default title

    # Heuristic for overall document title (often the largest, bold text on page 1)
    if structured_data and structured_data[0] and structured_data[0]['lines']:
        for line in structured_data[0]['lines']:
            if line['is_potential_heading'] and line['font_size_avg'] > 12.5 and line['is_bold_present']:
                # Prioritize a line that seems like a main title.
                # Avoid very short lines like just a number.
                if len(line['text']) > 10 and not re.match(r"^\d+$", line['text'].strip()):
                    document_title = line['text'].strip()
                    break # Assuming the first such line on page 1 is the main title

    # Collect all potential headings with their properties
    potential_headings = []
    for page_data in structured_data:
        for line_info in page_data["lines"]:
            text = line_info["text"].strip()
            # Skip empty lines, page numbers, or lines that are too short to be meaningful headings
            if not text or len(text) < 5 or re.match(r"^-?\d+\s*-$", text) or re.match(r"^\d+$", text):
                continue
            
            # Check if it's a potential heading and not a bullet point
            # Assuming bullet points are not main headings, but sub-headings might be.
            # Bullet point heuristic: starts with '•', '-', '*', 'o' followed by space
            if line_info["is_potential_heading"] and not re.match(r"^[\•\-\*o]\s+.*", text):
                potential_headings.append({
                    "text": text,
                    "page": page_data["page_number"],
                    "font_size_avg": line_info["font_size_avg"],
                    "is_bold_present": line_info["is_bold_present"],
                    "is_potential_heading": line_info["is_potential_heading"] # <--- ADD THIS LINE
                })

    # Deduplicate headings and assign levels
    seen_heading_texts = set()
    for heading in potential_headings:
        normalized_text = heading['text'].lower()
        if normalized_text in seen_heading_texts:
            continue
        seen_heading_texts.add(normalized_text)

        level = "H3" # Default to H3 if not classified higher

        # Heuristic 1: Numerical Headings (most reliable for H1/H2/H3)
        match_h1 = re.match(r"^\d+\.\s+.*", heading['text']) # e.g., "1. Introduction"
        match_h2 = re.match(r"^\d+\.\d+\s+.*", heading['text']) # e.g., "1.1 Subtitle"
        match_h3 = re.match(r"^\d+\.\d+\.\d+\s+.*", heading['text']) # e.g., "1.1.1 Sub-subtitle"

        if match_h1:
            level = "H1"
        elif match_h2:
            level = "H2"
        elif match_h3:
            level = "H3"
        # Heuristic 2: Non-numerical headings based on text patterns and font size/boldness
        else:
            # Common top-level sections (often bold and prominent)
            top_level_keywords = [
                "Introduction", "Overview of the Region", "Travel Tips", "Conclusion",
                "Marseille:", "Nice:", "Avignon:", "Aix-en-Provence:", "Toulouse:",
                "Montpellier:", "Perpignan:", "Arles:", "Carcassonne:",
                "Comprehensive Guide to Major Cities in the South of France" # Added the specific title from your PDF
            ]
            
            is_top_level_keyword = any(keyword.lower() in normalized_text for keyword in top_level_keywords)

            # Assign H1 if it's a strong potential heading, relatively short, and matches top-level keywords/patterns
            if heading['is_potential_heading'] and len(heading['text']) < 100: # This check is now safe
                if (is_top_level_keyword and heading['is_bold_present'] and heading['font_size_avg'] >= 12.0) or \
                   (heading['font_size_avg'] > 13.0 and heading['is_bold_present']): # Larger bold text
                    level = "H1"
                # Assign H2 for common sub-sections
                elif re.match(r"^(History|Key Attractions|Local Experiences|Cultural Highlights|Hidden Gems|Artistic Influence|Aerospace Industry|Student Life|Cultural Fusion|Best Time to Visit|Transportation|Language)", heading['text'], re.IGNORECASE) and heading['is_bold_present']:
                    level = "H2"
                elif heading['font_size_avg'] > 12.0 and heading['is_bold_present']: # If not caught by specific keywords, but still looks like a heading
                     level = "H2" # Default to H2 if strong potential heading but not H1

        document_outline.append({
            "level": level,
            "text": heading['text'],
            "page": heading['page']
        })

    return {
        "title": document_title,
        "outline": document_outline
    }


# --- Main execution block for single file outline extraction ---
if __name__ == "__main__":
    # Prompt user for PDF file path
    pdf_file_path = input("Enter the path to the PDF file (e.g., documents/South of France - Cities.pdf): ").strip()
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file '{pdf_file_path}' not found.")
        exit()

    output_outline_filename = "extracted_outline.json"

    print(f"Extracting outline from: {pdf_file_path}")
    outline_data = extract_document_outline(pdf_file_path)

    if outline_data["outline"]:
        print(f"Found {len(outline_data['outline'])} potential headings.")
    else:
        print("No headings identified based on current heuristics.")

    print(f"\nSaving extracted outline to {output_outline_filename}...")
    try:
        with open(output_outline_filename, "w", encoding="utf-8") as f:
            json.dump(outline_data, f, indent=2)
        print(f"Extracted outline saved to {output_outline_filename}")
    except Exception as e:
        print(f"Error saving extracted outline: {e}")

    # Optional: Print a snippet for quick review
    print("\n--- Extracted Outline (Snippet) ---")
    if outline_data["outline"]:
        for i, entry in enumerate(outline_data["outline"][:10]): # Print first 10 entries
            print(f"{i+1}. {entry['level']}: '{entry['text']}' (Page: {entry['page']})")
        if len(outline_data["outline"]) > 10:
            print(f"... and {len(outline_data['outline']) - 10} more entries.")
    else:
        print("Outline is empty.")