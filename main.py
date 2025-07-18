import os
import json
import fitz  # PyMuPDF

def analyze_pdf_structure(pdf_path):
    """Analyzes a PDF to extract its title and a hierarchical outline."""
    doc = fitz.open(pdf_path)
    outline = []
    title = ""

    text_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:  # Code for a text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        text_blocks.append({
                            "text": s["text"].strip(),
                            "size": s["size"],
                            "font": s["font"],
                            "bold": "bold" in s["font"].lower(),
                            "page": page_num + 1
                        })
    
    if not text_blocks:
        return {"title": "No text found", "outline": []}

    # Find the most common font size for body text
    font_sizes = [block['size'] for block in text_blocks if block['text']]
    if not font_sizes:
        return {"title": "No text found", "outline": []}
        
    body_size = max(set(font_sizes), key=font_sizes.count)
    
    # Filter for potential headings (larger than body text and often bold)
    potential_headings = [
        block for block in text_blocks 
        if block['size'] > body_size and block['text']
    ]
    
    # Find Title (largest font size, typically on the first page)
    if potential_headings:
        title_block = max(potential_headings, key=lambda x: x['size'])
        title = title_block['text']
        potential_headings.remove(title_block)

    # Group remaining headings by font size to determine H1, H2, H3
    heading_sizes = sorted(list(set([h['size'] for h in potential_headings])), reverse=True)
    
    level_map = {}
    if len(heading_sizes) > 0: level_map[heading_sizes[0]] = "H1"
    if len(heading_sizes) > 1: level_map[heading_sizes[1]] = "H2"
    if len(heading_sizes) > 2: level_map[heading_sizes[2]] = "H3"
    
    for head in potential_headings:
        if head['size'] in level_map:
            outline.append({
                "level": level_map[head['size']],
                "text": head['text'],
                "page": head['page']
            })
            
    doc.close()
    return {"title": title, "outline": outline}

def main():
    """Main function to process all PDFs in the input directory."""
    input_dir = "/app/input"
    output_dir = "/app/output"

    # for local testing comment out this portion
    # input_dir = "input"
    # output_dir = "output"

    print("Starting PDF processing...")
    
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
            
            result = analyze_pdf_structure(pdf_path)

            if not result:
                print(f"Failed to analyze {filename}.")
                continue
            if not result or not result.get("outline"):
                print(f"No valid structure found in {filename}.")
                continue
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    main()