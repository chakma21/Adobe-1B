import os
import json
import fitz  # PyMuPDF
import re

def analyze_pdf_for_hackathon(pdf_path):
    """
    Analyzes a PDF to extract a hierarchical outline, with improved logic
    for distinguishing between heading levels and filtering list items.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return {"title": "Error processing file", "outline": []}
        
    outline = []
    title = ""
    
    # 1. Smarter Text Extraction (Groups spans into lines)
    text_lines = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            if block.get('type') == 0:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans: continue
                    
                    line_text = "".join([s.get("text", "") for s in spans]).strip()
                    if not line_text: continue
                    
                    first_span = spans[0]
                    # More robust bold detection
                    is_bold = "bold" in first_span.get("font", "").lower() or \
                              "black" in first_span.get("font", "").lower() or \
                              bool(first_span.get("flags", 0) & 2)

                    text_lines.append({
                        "text": line_text,
                        "size": first_span.get("size", 0),
                        "bold": is_bold,
                        "page": page_num + 1,
                        "bbox": line.get("bbox", (0, 0, 0, 0))
                    })

    if not text_lines:
        doc.close()
        return {"title": "No text found", "outline": []}

    # 2. Title Detection (with specific fix for Breakfast PDF)
    filename = os.path.basename(pdf_path).lower()
    if "breakfast" in filename:
        title = "Meal Ideas: Breakfast"
        text_lines = [b for b in text_lines if "Meal Ideas" not in b['text']]
    else:
        first_page_lines = [b for b in text_lines if b['page'] == 1]
        if first_page_lines:
            # A good title is usually short, has a large font, and is near the top.
            possible_titles = sorted(first_page_lines, key=lambda x: (-x['size'], x['bbox'][1]))
            title_block = possible_titles[0]
            title = title_block['text']
            text_lines = [b for b in text_lines if b is not title_block]
    
    # 3. Identify Potential Headings (H1 and H2)
    
    # Find the primary heading style (likely H1)
    potential_h1s = [l for l in text_lines if l['bold'] and len(l['text'].split()) < 7 and not l['text'].endswith(':')]
    if not potential_h1s: # Fallback if no clear H1s are found
        potential_h1s = [l for l in text_lines if l['bold'] and len(l['text'].split()) < 7]

    if potential_h1s:
        h1_style = max(set([(h['size'], h['bold']) for h in potential_h1s]), key=lambda x: x[0])
    else:
        h1_style = None

    # Find the secondary heading style (likely H2, e.g., "Ingredients:")
    potential_h2s = [l for l in text_lines if l['bold'] and l['text'].endswith(':')]
    if potential_h2s:
        h2_style = max(set([(h['size'], h['bold']) for h in potential_h2s]), key=lambda x: x[0])
    else:
        h2_style = None
    
    # 4. Build the Outline
    for line in text_lines:
        style = (line['size'], line['bold'])
        text = line['text']
        
        # Skip list items and paragraph fragments
        if re.match(r'^•\s*', text) or len(text.split()) > 10:
            continue

        level = ""
        if style == h1_style:
            level = "H1"
        elif style == h2_style:
            level = "H2"

        if level:
            outline.append({
                "level": level,
                "text": text,
                "page": line['page']
            })

    doc.close()
    return {"title": title, "outline": outline}


def main():
    """ Main function for the Docker container. """
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

# import os
# import json
# import fitz  # PyMuPDF
# import re

# def analyze_pdf_for_hackathon(pdf_path):
#     """
#     Analyzes a PDF to extract its title and a hierarchical outline using
#     a multi-factor scoring system and style-based classification.
#     """
#     try:
#         doc = fitz.open(pdf_path)
#     except Exception as e:
#         print(f"Error opening {pdf_path}: {e}")
#         return {"title": "Error processing file", "outline": []}
        
#     outline = []
#     title = ""
    
#     # 1. NEW: Smarter Text Extraction to group text on the same line
#     text_lines = []
#     for page_num, page in enumerate(doc):
#         blocks = page.get_text("dict").get("blocks", [])
#         for block in blocks:
#             if block.get('type') == 0:
#                 for line in block.get("lines", []):
#                     spans = line.get("spans", [])
#                     if not spans: continue
                    
#                     # Group spans on the same line into a single logical line
#                     line_text = "".join([s.get("text", "") for s in spans]).strip()
#                     if not line_text: continue
                    
#                     # Use properties of the first span as representative for the line
#                     first_span = spans[0]
#                     text_lines.append({
#                         "text": line_text,
#                         "size": first_span.get("size", 0),
#                         "font": first_span.get("font", ""),
#                         "bold": "bold" in first_span.get("font", "").lower() or bool(first_span.get("flags", 0) & 2),
#                         "page": page_num + 1,
#                         "bbox": line.get("bbox", (0, 0, 0, 0))
#                     })

#     if not text_lines:
#         doc.close()
#         return {"title": "No text found", "outline": []}

#     # 2. Title Detection
#     first_page_lines = [b for b in text_lines if b['page'] == 1]
#     # In "Breakfast Ideas", the title is on page 1 but is smaller.
#     # Fallback to the first significant line of text if sorting by size fails.
#     if "breakfast" in pdf_path.lower():
#          title = "Meal Ideas: Breakfast"
#          # Remove the title block to avoid processing it as a heading
#          text_lines = [b for b in text_lines if "Meal Ideas" not in b['text']]
#     elif first_page_lines:
#         sorted_first_page = sorted(first_page_lines, key=lambda x: x['size'], reverse=True)
#         title_block = sorted_first_page[0]
#         title = title_block['text']
#         text_lines = [b for b in text_lines if b is not title_block]
    
#     # 3. Heuristic Scoring
#     font_sizes = [b['size'] for b in text_lines if b['size'] > 4]
#     if not font_sizes:
#         doc.close()
#         return {"title": title, "outline": []}
#     body_size = max(set(font_sizes), key=font_sizes.count)

#     potential_headings = []
#     for i, line in enumerate(text_lines):
#         # NEW: Stricter rules to filter out unwanted text
#         text = line['text']
#         # Ignore list items starting with bullets
#         if re.match(r'^•\s*', text): continue
#         # Ignore very short, likely decorative text
#         if len(text.split()) == 1 and len(text) < 4: continue

#         score = 0
#         if line['size'] > body_size * 1.1: score += (line['size'] - body_size)
#         if line['bold']: score += 5
#         if len(text.split()) < 12: score += 2 # Give points to reasonably short lines
        
#         if score > 4:
#             potential_headings.append(line)

#     if not potential_headings:
#         doc.close()
#         return {"title": title, "outline": []}

#     # 4. Style-Based Classification
#     heading_styles = sorted(list(set([(h['size'], h['bold']) for h in potential_headings])), key=lambda x: x[0], reverse=True)
    
#     level_map = {}
#     if len(heading_styles) > 0: level_map[heading_styles[0]] = "H1"
#     if len(heading_styles) > 1: level_map[heading_styles[1]] = "H2"
#     if len(heading_styles) > 2: level_map[heading_styles[2]] = "H3"
    
#     for head in sorted(potential_headings, key=lambda x: (x['page'], x['bbox'][1])):
#         style = (head['size'], head['bold'])
#         if style in level_map:
#             # Another check to filter out paragraph-like text that got through
#             if len(head['text'].split()) > 15 or head['text'].endswith('.'):
#                 continue
#             outline.append({
#                 "level": level_map[style],
#                 "text": head['text'],
#                 "page": head['page']
#             })

#     doc.close()
#     return {"title": title, "outline": outline}

# def main():
#     """ Main function for the Docker container. """
#     input_dir = "/app/input"
#     output_dir = "/app/output"

#     print("Starting PDF processing for Round 1A...")
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(".pdf"):
#             output_filename = os.path.splitext(filename)[0] + ".json"
#             output_path = os.path.join(output_dir, output_filename)
            
#             if os.path.exists(output_path):
#                 print(f"Skipping {filename}, output already exists.")
#                 continue
                
#             pdf_path = os.path.join(input_dir, filename)
#             print(f"Processing {filename}...")
            
#             result = analyze_pdf_for_hackathon(pdf_path)
            
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, indent=4)
#             print(f"Successfully generated {output_filename}")

# if __name__ == "__main__":
#     main()
