import json
import os
from datetime import datetime
import re
import fitz  # PyMuPDF library for PDF processing
import numpy as np # For numerical operations, especially with embeddings
from typing import List, Dict, Any

# Import for semantic matching
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: Missing required libraries. Please install them using:")
    print("pip install sentence-transformers scikit-learn PyMuPDF numpy")
    exit(1)

# --- CORRECTED MODEL PATH FOR DOCKER ---
# This absolute path points to where the model is saved inside the Docker image.
MODEL_PATH = '/model'

# Attempt to load the SentenceTransformer model
try:
    model = SentenceTransformer(MODEL_PATH)
    print(f"Successfully loaded SentenceTransformer model from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load SentenceTransformer model from {MODEL_PATH}.")
    print(f"Detailed error: {e}")
    exit(1) # Exit if model loading fails, as it's critical

def get_text_embedding(text: str) -> np.ndarray:
    """Generates an embedding vector for a given text using the pre-loaded model."""
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(text, convert_to_tensor=False)

def calculate_semantic_relevance_score(text1: str, text2: str) -> float:
    """
    Calculates a semantic relevance score (cosine similarity) between two texts.
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    embedding1 = get_text_embedding(text1)
    embedding2 = get_text_embedding(text2)

    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    normalized_similarity = (similarity + 1) / 2
    return normalized_similarity


def extract_structured_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text blocks with basic styling information from a PDF.
    """
    document_pages_structured_content = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_blocks_on_page = page.get_text("dict")["blocks"]
            
            page_content_lines = []
            for block in text_blocks_on_page:
                if block['type'] == 0: # Text block
                    for line in block['lines']:
                        line_text = ""
                        spans_info = []
                        for span in line['spans']:
                            line_text += span['text']
                            is_span_bold = bool(span['flags'] & 2) or \
                                           ("bold" in span['font'].lower()) or \
                                           ("heavy" in span['font'].lower()) or \
                                           ("black" in span['font'].lower()) or \
                                           ("extrabold" in span['font'].lower()) 

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
                            
                            if (avg_size >= 12.5 and is_bold_present) or (avg_size > 14.0):
                                is_potential_heading = True

                        page_content_lines.append({
                            "text": line_text.strip(),
                            "is_potential_heading": is_potential_heading,
                            "font_size_avg": avg_size,
                            "is_bold_present": is_bold_present
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


def process_document_collection(input_json_data: Dict[str, Any], documents_dir: str) -> List[Dict[str, Any]]:
    """
    Processes a collection of PDFs to identify and extract candidate sections.
    """
    all_candidate_sections = [] 
    
    for doc_info in input_json_data["documents"]:
        filename = doc_info["filename"]
        pdf_path = os.path.join(documents_dir, filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: Document not found at {pdf_path}. Skipping '{filename}'.")
            continue 

        print(f"Extracting text and identifying sections from: {filename}")
        pages_structured_text = extract_structured_text_from_pdf(pdf_path)

        if not pages_structured_text:
            print(f"No text extracted from {filename}.")
            continue

        current_section_title = doc_info.get('title', os.path.splitext(filename)[0])
        current_section_page = 1
        current_section_content_buffer = []
        seen_section_titles_in_doc = set() 

        for page_data in pages_structured_text:
            page_number = page_data["page_number"]
            for i, line_info in enumerate(page_data["lines"]):
                text = line_info["text"].strip()

                if not text or len(text) < 5 or re.match(r"^-?\s*\d+\s*-$", text) or re.match(r"^\s*\d+\s*$", text):
                    continue

                is_new_section_candidate = False
                normalized_text = text.lower()

                if line_info["is_potential_heading"] and \
                   len(text) < 100 and \
                   not re.match(r"^(?:\s*[\•\-\*o]|\d+\.)\s+.*", text) and \
                   not text.endswith(('.', '!', '?')) and \
                   normalized_text not in seen_section_titles_in_doc:
                    
                    is_new_section_candidate = True

                if is_new_section_candidate:
                    if current_section_content_buffer:
                        all_candidate_sections.append({
                            "document": filename,
                            "section_title": current_section_title,
                            "page_number": current_section_page,
                            "full_section_content": " ".join(current_section_content_buffer).strip()
                        })
                        
                    current_section_title = text
                    current_section_page = page_number
                    current_section_content_buffer = []
                    seen_section_titles_in_doc.add(normalized_text)
                else:
                    current_section_content_buffer.append(text)
        
        if current_section_content_buffer:
            all_candidate_sections.append({
                "document": filename,
                "section_title": current_section_title,
                "page_number": current_section_page,
                "full_section_content": " ".join(current_section_content_buffer).strip()
            })
    
    return all_candidate_sections


def identify_document_sections_with_semantic_ranking(all_candidate_sections: List[Dict[str, Any]], job_to_be_done_query: str) -> List[Dict[str, Any]]:
    """
    Calculates semantic relevance for all candidate sections and ranks them.
    """
    ranked_sections = []
    
    for section_data in all_candidate_sections:
        full_text_for_ranking = section_data["section_title"] + " " + section_data["full_section_content"]
        
        relevance_score = calculate_semantic_relevance_score(full_text_for_ranking, job_to_be_done_query)
        
        ranked_sections.append({
            "document": section_data["document"],
            "section_title": section_data["section_title"],
            "importance_rank_score": float(relevance_score),
            "page_number": section_data["page_number"],
            "full_section_content": section_data["full_section_content"]
        })

    ranked_sections.sort(key=lambda x: x["importance_rank_score"], reverse=True)
    return ranked_sections

def summarize_section_content(section_full_text: str, job_to_be_done_query: str, max_length_sentences: int = 5) -> str:
    """
    Generates a concise summary of section content by extracting relevant sentences.
    """
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', section_full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""
    
    if len(sentences) <= max_length_sentences:
        return " ".join(sentences)

    job_embedding = get_text_embedding(job_to_be_done_query).reshape(1, -1)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=False)
    sentence_scores = cosine_similarity(job_embedding, sentence_embeddings)[0]

    selected_indices = []
    
    if len(sentences[0]) > 30 and sentences[0].count(' ') > 3:
        selected_indices.append(0)

    scored_sentences = [(score, idx) for idx, score in enumerate(sentence_scores)]
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    for score, idx in scored_sentences:
        if len(selected_indices) >= max_length_sentences:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    selected_indices.sort()
    
    summarized_sentences = [sentences[i] for i in selected_indices]
    
    return " ".join(summarized_sentences)


# --- Main execution block ---
if __name__ == "__main__":
    input_json_filename = "input_payload.json"
    documents_directory = "documents"
    output_filename = "processed_document_output.json"

    print(f"Loading input payload from {input_json_filename}...")
    if not os.path.exists(input_json_filename):
        print(f"ERROR: Input JSON file '{input_json_filename}' not found.")
        exit(1)

    try:
        with open(input_json_filename, 'r', encoding='utf-8') as f:
            input_payload = json.load(f)
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_json_filename}': {e}")
        exit(1)

    persona_role = input_payload["persona"]["role"]
    job_task = input_payload["job_to_be_done"]["task"]
    job_to_be_done_query = f"Persona: {persona_role}. Task: {job_task}"
    print(f"Job to be done query: '{job_to_be_done_query}'")

    if not os.path.exists(documents_directory):
        print(f"ERROR: Documents directory '{documents_directory}' not found.")
        exit(1)
    print(f"Using documents from directory: {documents_directory}")

    print("\nStarting document text extraction and section identification...")
    all_candidate_sections = process_document_collection(input_payload, documents_dir=documents_directory)
    print(f"Initial section identification complete. Found {len(all_candidate_sections)} candidate sections.")

    print(f"\nRanking candidate sections by relevance...")
    identified_and_ranked_sections = identify_document_sections_with_semantic_ranking(all_candidate_sections, job_to_be_done_query)

    final_output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_payload["documents"]],
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    num_top_sections = min(5, len(identified_and_ranked_sections)) 
    seen_analysis_entries = set() 

    print(f"\nPopulating top {num_top_sections} sections and generating analysis...")
    for i in range(num_top_sections):
        section = identified_and_ranked_sections[i]
        
        final_output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": i + 1,
            "page_number": section["page_number"]
        })

        full_section_content = section.get("full_section_content", "")
        refined_text = "No content available for summarization."
        if full_section_content.strip():
            refined_text = summarize_section_content(full_section_content, job_to_be_done_query)
        
        entry_key = (section["document"], section["page_number"], refined_text)
        if entry_key not in seen_analysis_entries:
            final_output["subsection_analysis"].append({
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"]
            })
            seen_analysis_entries.add(entry_key)

    print(f"Subsection analysis complete. Generated {len(final_output['subsection_analysis'])} entries.")

    print(f"\nSaving final output to {output_filename}...")
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"Full processed output saved to {output_filename}")
    except Exception as e:
        print(f"ERROR: Could not save final output: {e}")
        exit(1)