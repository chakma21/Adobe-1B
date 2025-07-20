import os
import json
import fitz  # PyMuPDF
import re

def analyze_pdf_for_hackathon(pdf_path):
    """
    Analyzes a PDF to extract its title and a hierarchical outline using
    a multi-factor scoring system, formatted for the hackathon output.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return {"title": "Error processing file", "outline": []}
        
    outline = []
    title = ""
    
    # 1. Reliable Text Extraction (from your new script's logic)
    text_blocks = []
    for page_num, page in enumerate(doc):
        # Using page.get_text("dict") is a robust way to get all text spans
        page_blocks = page.get_text("dict").get("blocks", [])
        for block in page_blocks:
            if block.get('type') == 0:  # This is a text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        
                        text_blocks.append({
                            "text": text,
                            "size": span.get("size", 0),
                            "font": span.get("font", ""),
                            # Using flags is a more reliable way to detect bold
                            "bold": bool(span.get("flags", 0) & 2),
                            "page": page_num + 1,
                            "bbox": span.get("bbox", (0, 0, 0, 0))
                        })

    if not text_blocks:
        doc.close()
        return {"title": "No text found", "outline": []}

    # 2. Heuristic Analysis (using our advanced scoring system)
    font_sizes = [b['size'] for b in text_blocks]
    body_size = max(set(font_sizes), key=font_sizes.count)

    scored_blocks = []
    for i, block in enumerate(text_blocks):
        score = 0
        if block['size'] > body_size: score += (block['size'] - body_size) * 1.5
        if block['bold']: score += 5
        if i > 0 and text_blocks[i-1]['page'] == block['page']:
            space_above = block['bbox'][1] - text_blocks[i-1]['bbox'][3]
            if space_above > (block['size'] * 0.5): score += 4
        else: score += 2 # Boost for being first on a page
        if len(block['text'].split()) < 10: score += 3
        if block['text'].isupper() and len(block['text']) > 1: score += 4
        if re.match(r'^\d+(\.\d+)*\s|\b(Chapter|Section|Introduction|Conclusion)\b', block['text'], re.IGNORECASE): score += 10
        if score > 0:
            block['score'] = score
            scored_blocks.append(block)

    # 3. Filtering and Classification for Hackathon Output
    heading_threshold = 8
    potential_headings = sorted([b for b in scored_blocks if b.get('score', 0) > heading_threshold], key=lambda x: x.get('score', 0), reverse=True)

    if not potential_headings:
        doc.close()
        return {"title": "", "outline": []}

    # Title detection
    first_page_headings = [h for h in potential_headings if h['page'] == 1]
    title_block = max(first_page_headings, key=lambda x: x.get('score', 0)) if first_page_headings else potential_headings[0]
    title = title_block['text']
    potential_headings = [h for h in potential_headings if h is not title_block]

    # Classify H1, H2, H3 based on score
    if potential_headings:
        max_score = potential_headings[0]['score']
        h1_thresh = max_score * 0.8
        h2_thresh = max_score * 0.6
        
        for head in sorted(potential_headings, key=lambda x: (x['page'], x['bbox'][1])):
            level = "H3" # Default to H3 for headings that pass threshold but have lower scores
            score = head.get('score', 0)
            if score >= h1_thresh: level = "H1"
            elif score >= h2_thresh: level = "H2"
            
            outline.append({"level": level, "text": head['text'], "page": head['page']})

    doc.close()
    return {"title": title, "outline": outline}

def main():
    """
    Main function for the Docker container. Processes all PDFs from /app/input
    and writes JSON outlines to /app/output.
    """
    input_dir = "/app/input"
    output_dir = "/app/output"

    print("Starting PDF processing for Round 1A...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(output_path):
                print(f"Skipping {filename}, output already exists.")
                continue
                
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            
            result = analyze_pdf_for_hackathon(pdf_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    main()