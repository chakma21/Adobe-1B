import fitz # PyMuPDF library for PDF processing
import json # For working with JSON data
import os # For interacting with the operating system (e.g., file paths, directory creation)
from datetime import datetime # For generating timestamps
import re # For regular expressions, useful for pattern matching in text

# Import for semantic matching
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Initialize the Sentence-Transformer Model ---
# This model is loaded once at the beginning of your script.
# 'all-MiniLM-L6-v2' is a good general-purpose model, small and fast.
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an internet connection to download the model.")
    print("If issues persist, try running: pip install sentence-transformers")
    exit()

def get_text_embedding(text):
    """Generates an embedding vector for a given text using the pre-loaded model."""
    # Ensure text is not empty, as model.encode might error on empty string
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension()) # Return zero vector for empty text
    return model.encode(text, convert_to_tensor=False) # convert_to_tensor=False returns numpy array

def calculate_semantic_relevance_score(section_content_text, job_to_be_done_text):
    """
    Calculates a semantic relevance score between section content and the job to be done.
    Scores range from 0 to 1.
    """
    if not section_content_text.strip():
        return 0.0

    job_embedding = get_text_embedding(job_to_be_done_text)
    section_embedding = get_text_embedding(section_content_text)

    # Reshape for cosine_similarity if only one sample
    job_embedding = job_embedding.reshape(1, -1)
    section_embedding = section_embedding.reshape(1, -1)

    # Cosine similarity ranges from -1 to 1. Normalize to 0-1 range for easier interpretation.
    similarity = cosine_similarity(job_embedding, section_embedding)[0][0]
    normalized_similarity = (similarity + 1) / 2 # Normalize to 0-1

    return normalized_similarity

def extract_structured_text_from_pdf(pdf_path):
    """
    Extracts text blocks with basic styling information (font size, bold status)
    from a single PDF document.
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
                            })
                        
                        is_potential_heading = False
                        avg_size = 0
                        is_bold_present = False

                        if spans_info:
                            avg_size = sum([s['size'] for s in spans_info]) / len(spans_info)
                            is_bold_present = any(s['is_bold'] for s in spans_info)
                            
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

def identify_document_sections_with_semantic_ranking(extracted_structured_data, job_to_be_done_query):
    """
    Identifies main sections within each document based on the structured text,
    leveraging font information and semantic ranking.
    """
    extracted_sections = []
    
    for doc_data in extracted_structured_data["extracted_documents_structured_text"]:
        filename = doc_data["filename"]
        seen_section_titles_in_doc = set()

        for page_data in doc_data["pages"]:
            page_number = page_data["page_number"]
            for i, line_info in enumerate(page_data["lines"]):
                text = line_info["text"].strip()

                if not text or len(text) < 5:
                    continue

                is_likely_section_title = False
                # Refined heuristic: Combine conditions to make it more robust
                if line_info["is_potential_heading"] and len(text) < 100: 
                    normalized_text = text.lower()
                    if normalized_text not in seen_section_titles_in_doc:
                        # Stronger heuristic: all caps and short, or Title Case, or starting with common section words
                        if (text.isupper() and len(text.split()) < 10) or \
                           (text == text.title() and len(text.split()) < 15) or \
                           (re.match(r"^(Introduction|Chapter|Section|Overview|Key|Summary|Conclusion|Tips|Things to Do|Coastal|Culinary|Nightlife)", text, re.IGNORECASE)):
                            is_likely_section_title = True
                        elif line_info["is_bold_present"] and line_info["font_size_avg"] >= 12.0:
                             is_likely_section_title = True

                if is_likely_section_title:
                    # Collect the full content of the current section
                    # Iterate through subsequent lines until the next potential heading or end of page/document
                    section_content_lines = []
                    # Start from the line after the current heading
                    for j in range(i + 1, len(page_data["lines"])):
                        next_line = page_data["lines"][j]
                        if next_line["is_potential_heading"]:
                            break # Stop if next heading is found
                        section_content_lines.append(next_line["text"])
                    
                    # Combine the heading text with its content for the full section text
                    section_full_text = text + " " + " ".join(section_content_lines).strip()

                    # Calculate semantic relevance score
                    relevance_score = calculate_semantic_relevance_score(section_full_text, job_to_be_done_query)
                    relevance_score = float(relevance_score) # IMPORTANT FIX: Convert numpy.float32 to standard float

                    extracted_sections.append({
                        "document": filename,
                        "section_title": text,
                        "importance_rank": relevance_score, # Assign the semantic relevance score
                        "page_number": page_number,
                        "full_section_content": section_full_text # Store full content for later summarization
                    })
                    seen_section_titles_in_doc.add(normalized_text)

    # Sort sections by importance_rank in descending order
    extracted_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
    return extracted_sections
    
def summarize_section_content(section_full_text, job_to_be_done_query, max_length_sentences=3):
    """
    Generates a concise summary of the section content based on relevance to the job
    and by extracting the most relevant sentences.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', section_full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""
    if len(sentences) <= max_length_sentences: # If few sentences, return all
        return " ".join(sentences)

    job_embedding = get_text_embedding(job_to_be_done_query).reshape(1, -1)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=False)

    # Calculate similarity of each sentence to the job query
    sentence_scores = cosine_similarity(job_embedding, sentence_embeddings)[0]

    # Get indices of top N sentences
    top_sentence_indices = np.argsort(sentence_scores)[::-1][:max_length_sentences]
    
    # Sort indices to maintain original order of sentences in the summary
    top_sentence_indices.sort()
    
    summarized_sentences = [sentences[i] for i in top_sentence_indices]
    
    return " ".join(summarized_sentences)


# --- Main execution block ---
if __name__ == "__main__":
    input_json_filename = "input_payload.json"
    documents_directory = "documents"
    output_filename = "processed_document_output.json"
    
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

    job_to_be_done_query = input_payload["job_to_be_done"]["task"] # Extract the job to be done

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


    # --- Step 4: Identify Sections and Rank them based on Semantic Relevance ---
    print(f"\nIdentifying sections and ranking by relevance to: '{job_to_be_done_query}'...")
    identified_sections = identify_document_sections_with_semantic_ranking(extracted_structured_data, job_to_be_done_query)
    print("Section identification and ranking complete.")

    # --- Step 5: Prepare Final Output Structure (including only top 5 sections) ---
    final_output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in extracted_structured_data["metadata"]["input_documents"]],
            "persona": extracted_structured_data["metadata"]["persona"]["role"],
            "job_to_be_done": extracted_structured_data["metadata"]["job_to_be_done"]["task"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": identified_sections[:5], # Take only the top 5 ranked sections
        "subsection_analysis": [] 
    }

    # --- Step 6: Generate Refined Subsection Analysis for the Top 5 Sections ---
    print("\nGenerating refined subsection analysis for top 5 sections...")
    for section in final_output["extracted_sections"]:
        document_filename = section["document"]
        section_title = section["section_title"]
        page_number = section["page_number"]
        full_section_content = section.get("full_section_content", "") # Get the full content stored earlier

        if full_section_content.strip(): # Only summarize if there's content
            refined_text = summarize_section_content(full_section_content, job_to_be_done_query)
            final_output["subsection_analysis"].append({
                "document": document_filename,
                "section_title": section_title, # Added section title to subsection_analysis for clarity
                "refined_text": refined_text,
                "page_number": page_number
            })
        else:
            final_output["subsection_analysis"].append({
                "document": document_filename,
                "section_title": section_title,
                "refined_text": "No content available for summarization.",
                "page_number": page_number
            })
    print("Subsection analysis complete.")

    # --- Step 7: Save Final Output to JSON File ---
    print(f"\nSaving full processed output to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"Full processed output saved to {output_filename}")

    # Optional: Print a snippet of the identified sections for quick review
    print("\n--- Identified Sections (Snippet) ---")
    if final_output["extracted_sections"]:
        for i, section in enumerate(final_output["extracted_sections"]): # Now prints all top 5
            print(f"{i+1}. Document: {section['document']}, Section: '{section['section_title']}', Rank: {section['importance_rank']:.4f}, Page: {section['page_number']}")
    else:
        print("No sections identified.")

    print("\n--- Refined Subsection Analysis (Snippet) ---")
    if final_output["subsection_analysis"]:
        for i, analysis in enumerate(final_output["subsection_analysis"]):
            print(f"{i+1}. Document: {analysis['document']}, Section: '{analysis['section_title']}', Page: {analysis['page_number']}")
            print(f"   Refined Text: {analysis['refined_text'][:150]}...") # Print first 150 chars
    else:
        print("No subsection analysis generated.")