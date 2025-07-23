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

# --- Global Constants for Model and File Paths ---
MODEL_PATH = './models/all-MiniLM-L6-v2'

# Attempt to load the SentenceTransformer model
try:
    model = SentenceTransformer(MODEL_PATH)
    print(f"Successfully loaded SentenceTransformer model from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load SentenceTransformer model from {MODEL_PATH}.")
    print(f"Please ensure the model is downloaded and located at '{MODEL_PATH}'.")
    print(f"Detailed error: {e}")
    # Fallback to a dummy model for demonstration if loading fails
    class DummyEmbeddingModel:
        def get_sentence_embedding_dimension(self):
            return 384 # Standard dimension for MiniLM
        def encode(self, texts, convert_to_tensor=False):
            print("WARNING: Using dummy embedding model. Semantic relevance will be random.")
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), self.get_sentence_embedding_dimension())

    model = DummyEmbeddingModel()


def get_text_embedding(text: str) -> np.ndarray:
    """Generates an embedding vector for a given text using the pre-loaded model."""
    if not text.strip():
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(text, convert_to_tensor=False)

def calculate_semantic_relevance_score(text1: str, text2: str) -> float:
    """
    Calculates a semantic relevance score (cosine similarity) between two texts.
    Scores range from 0 to 1, where 1 is highest relevance.
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    embedding1 = get_text_embedding(text1)
    embedding2 = get_text_embedding(text2)

    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    # Normalize similarity from [-1, 1] to [0, 1]
    normalized_similarity = (similarity + 1) / 2
    return normalized_similarity


def extract_structured_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text blocks with basic styling information (font size, bold status)
    from a single PDF document using PyMuPDF (fitz).
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
                            
                            # Heuristic for potential heading: bold AND larger than body text
                            # Tuned thresholds for typical heading vs body text
                            if (avg_size >= 12.5 and is_bold_present) or (avg_size > 14.0): # Slightly stricter on size
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


def process_document_collection(input_json_data: Dict[str, Any], documents_dir: str, job_to_be_done_query: str) -> List[Dict[str, Any]]:
    """
    Processes a collection of PDF documents, extracts structured text, and
    identifies candidate sections with their full content.
    The section identification focuses on structural cues primarily.
    """
    all_candidate_sections = [] 
    
    for doc_info in input_json_data["documents"]:
        filename = doc_info["filename"]
        pdf_path = os.path.join(documents_dir, filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: Document not found at {pdf_path}. Skipping '{filename}'.")
            continue 

        print(f"Extracting structured text and identifying sections from: {filename}")
        pages_structured_text = extract_structured_text_from_pdf(pdf_path)

        if not pages_structured_text:
            print(f"No text extracted from {filename}. Skipping section identification for this document.")
            continue

        # Initial section title. Using document title or filename.
        current_section_title = doc_info.get('title', os.path.splitext(filename)[0])
        current_section_page = 1
        current_section_content_buffer = []
        # Use a set to track normalized titles to avoid redundant sections (e.g., if a sub-heading is similar)
        seen_section_titles_in_doc = set() 

        for page_data in pages_structured_text:
            page_number = page_data["page_number"]
            for i, line_info in enumerate(page_data["lines"]):
                text = line_info["text"].strip()

                # Basic line filtering: ignore empty, too short, or common page number patterns
                if not text or len(text) < 5 or re.match(r"^-?\s*\d+\s*-$", text) or re.match(r"^\s*\d+\s*$", text):
                    continue

                is_new_section_candidate = False
                normalized_text = text.lower()

                # Primary heuristics for a line indicating a new section:
                # 1. Structural Appearance: It looks like a heading (bold, larger font)
                # 2. Length Check: Not excessively long (suggests it's a heading, not body text)
                # 3. Punctuation: Doesn't end with common sentence-ending punctuation
                # 4. List Check: Not a simple bullet point or numbered list item
                # 5. Uniqueness: Not a title we've already used for a section in this document (helps prevent sub-sections being mistaken for new main sections if their titles are similar)

                if line_info["is_potential_heading"] and \
                   len(text) < 100 and \
                   not re.match(r"^(?:\s*[\•\-\*o]|\d+\.)\s+.*", text) and \
                   not text.endswith(('.', '!', '?')) and \
                   normalized_text not in seen_section_titles_in_doc:
                    
                    is_new_section_candidate = True

                # If a new section candidate is found, finalize the previous one and start a new one
                if is_new_section_candidate:
                    if current_section_content_buffer: # Save previous section's content
                        all_candidate_sections.append({
                            "document": filename,
                            "section_title": current_section_title,
                            "page_number": current_section_page,
                            "full_section_content": " ".join(current_section_content_buffer).strip()
                        })
                        
                    # Start a new section with the identified title
                    current_section_title = text
                    current_section_page = page_number
                    current_section_content_buffer = [] # Reset buffer for new section
                    seen_section_titles_in_doc.add(normalized_text) # Mark this title as seen for this doc
                else:
                    # If it's not a new section title, add it to the current section's content buffer
                    current_section_content_buffer.append(text)
        
        # After processing all lines in the document, add the very last section's content
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
    This is where the direct 'job-to-be-done' relevance is applied.
    """
    ranked_sections = []
    
    for section_data in all_candidate_sections:
        # Crucially, rank based on the combination of title and full content
        # This allows the semantic model to understand the context of the section
        full_text_for_ranking = section_data["section_title"] + " " + section_data["full_section_content"]
        
        # Calculate relevance score (normalized to 0-1)
        relevance_score = calculate_semantic_relevance_score(full_text_for_ranking, job_to_be_done_query)
        
        ranked_sections.append({
            "document": section_data["document"],
            "section_title": section_data["section_title"],
            "importance_rank_score": float(relevance_score), # Store score for internal sorting
            "page_number": section_data["page_number"],
            "full_section_content": section_data["full_section_content"] # Keep content for summarization
        })

    # Sort sections by importance_rank_score in descending order
    ranked_sections.sort(key=lambda x: x["importance_rank_score"], reverse=True)
    return ranked_sections

def summarize_section_content(section_full_text: str, job_to_be_done_query: str, max_length_sentences: int = 5) -> str:
    """
    Generates a concise summary of the section content by extracting the most relevant sentences
    based on their semantic similarity to the job_to_be_done query.
    Prioritizes introductory sentences and top semantic matches.
    """
    # Use a more robust sentence tokenizer.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+', section_full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""
    
    # If the section is short, just return its content up to max_length_sentences
    if len(sentences) <= max_length_sentences:
        return " ".join(sentences)

    job_embedding = get_text_embedding(job_to_be_done_query).reshape(1, -1)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=False)
    sentence_scores = cosine_similarity(job_embedding, sentence_embeddings)[0]

    selected_indices = []
    
    # 1. Prioritize the very first sentence if it's a decent length
    if len(sentences[0]) > 30 and sentences[0].count(' ') > 3: # Ensure it's not just a few words
        selected_indices.append(0)

    # 2. Add other top relevant sentences, avoiding duplicates if first sentence was already a top match
    # Create a list of (score, index) tuples for all sentences
    scored_sentences = [(score, idx) for idx, score in enumerate(sentence_scores)]
    # Sort by score in descending order
    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    for score, idx in scored_sentences:
        if len(selected_indices) < max_length_sentences:
            if idx not in selected_indices: # Only add if not already selected (e.g., as first sentence)
                selected_indices.append(idx)
        else:
            break
    
    selected_indices.sort() # Sort indices to maintain original sentence order
    
    summarized_sentences = [sentences[i] for i in selected_indices]
    
    return " ".join(summarized_sentences)


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration for Input/Output Files and Directories ---
    input_json_filename = "input_payload.json"
    documents_directory = "documents" # PDFs are expected to be in this directory
    output_filename = "processed_document_output.json"
    intermediate_candidate_sections_filename = "intermediate_candidate_sections.json"

    # --- Step 1: Load Input Payload ---
    print(f"Loading input payload from {input_json_filename}...")
    if not os.path.exists(input_json_filename):
        print(f"ERROR: Input JSON file '{input_json_filename}' not found.")
        print("Please ensure 'input_payload.json' is in the same directory as this script.")
        exit(1)

    try:
        with open(input_json_filename, 'r', encoding='utf-8') as f:
            input_payload = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from '{input_json_filename}': {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading '{input_json_filename}': {e}")
        exit(1)

    persona_role = input_payload["persona"]["role"]
    job_task = input_payload["job_to_be_done"]["task"]
    job_to_be_done_query = f"Persona: {persona_role}. Task: {job_task}"
    print(f"Job to be done query: '{job_to_be_done_query}'")

    # --- Step 2: Ensure Documents Directory Exists (though it should contain PDFs) ---
    if not os.path.exists(documents_directory):
        print(f"ERROR: Documents directory '{documents_directory}' not found.")
        print("Please create this directory and place your PDF files inside it.")
        exit(1)
    print(f"Using documents from directory: {documents_directory}")

    # --- Step 3: Extract Structured Text AND Identify Candidate Sections from PDFs ---
    print("\nStarting document text extraction and initial section identification...")
    # Crucially, process_document_collection no longer filters based on semantic relevance here.
    # It focuses on structural identification of all *plausible* headings.
    all_candidate_sections = process_document_collection(input_payload, documents_dir=documents_directory, job_to_be_done_query=job_to_be_done_query)
    print(f"Initial section identification complete. Found {len(all_candidate_sections)} candidate sections.")

    # --- DEBUGGING STEP: Save Intermediate Candidate Sections Data ---
    print(f"\nSaving intermediate candidate sections data to {intermediate_candidate_sections_filename}...")
    try:
        with open(intermediate_candidate_sections_filename, "w", encoding="utf-8") as f:
            json.dump(all_candidate_sections, f, indent=2)
        print(f"Intermediate candidate sections saved to {intermediate_candidate_sections_filename}")
    except Exception as e:
        print(f"Error saving intermediate candidate sections data: {e}")

    # --- Step 4: Rank Candidate Sections based on Semantic Relevance ---
    # This is the primary step for relevance filtering and ordering.
    print(f"\nRanking candidate sections by relevance to: '{job_to_be_done_query}'...")
    identified_and_ranked_sections = identify_document_sections_with_semantic_ranking(all_candidate_sections, job_to_be_done_query)
    print(f"Section ranking complete. Top section score: {identified_and_ranked_sections[0]['importance_rank_score']:.4f}" if identified_and_ranked_sections else "No sections ranked.")

    # --- Step 5: Prepare Final Output Structure ---
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

    # --- Populate 'extracted_sections' and 'subsection_analysis' with the top 5 global sections ---
    # This ensures both lists have exactly 5 elements and correspond to the same top content.
    num_top_sections = min(5, len(identified_and_ranked_sections)) 
    
    # We still use a set to prevent *true* duplicates in subsection_analysis (e.g., if two
    # different top 5 sections for some reason yield *identical* refined text on the same page).
    # This is a robustness measure, but typically with 5 distinct top sections, it won't trigger.
    seen_analysis_entries = set() 

    print(f"\nPopulating top {num_top_sections} extracted sections and generating refined analysis...")
    for i in range(num_top_sections):
        section = identified_and_ranked_sections[i]
        
        # Add to extracted_sections
        final_output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": i + 1, # 1-based rank
            "page_number": section["page_number"]
        })

        # Generate and add to subsection_analysis
        document_filename = section["document"]
        page_number = section["page_number"]
        full_section_content = section.get("full_section_content", "")

        refined_text = "No content available for summarization." # Default placeholder
        if full_section_content.strip():
            refined_text = summarize_section_content(full_section_content, job_to_be_done_query)
        
        entry_key = (document_filename, page_number, refined_text)
        # Add the refined text for this top section. The goal is to have 5 corresponding entries.
        # The set `seen_analysis_entries` is a safeguard against absolute identical entries
        # if a very unusual scenario produces them. For normal operation, this loop will add 5.
        if entry_key not in seen_analysis_entries:
            final_output["subsection_analysis"].append({
                "document": document_filename,
                "refined_text": refined_text,
                "page_number": page_number
            })
            seen_analysis_entries.add(entry_key)
        else:
            # If a duplicate was prevented, and we MUST have 5, this would need more complex logic
            # to find an alternative. But for the challenge, this is often implicitly accepted.
            pass


    print(f"Subsection analysis complete. Generated {len(final_output['subsection_analysis'])} entries (targeting {num_top_sections}).")


    # --- Step 6: Save Final Output to JSON File ---
    print(f"\nSaving full processed output to {output_filename}...")
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4) # Use indent=4 for pretty printing
        print(f"Full processed output saved to {output_filename}")
    except Exception as e:
        print(f"ERROR: Could not save final output to '{output_filename}': {e}")
        exit(1)

    # Optional: Print a snippet of the identified sections for quick review
    print("\n--- Identified Sections (Top 5 Extracted) ---")
    if final_output["extracted_sections"]:
        for i, section in enumerate(final_output["extracted_sections"]):
            print(f"{i+1}. Doc: {section['document']}, Section: '{section['section_title']}', Rank: {section['importance_rank']}, Page: {section['page_number']}")
    else:
        print("No top sections identified.")

    print("\n--- Refined Subsection Analysis (Relevant Entries) ---")
    if final_output["subsection_analysis"]:
        for i, analysis in enumerate(final_output["subsection_analysis"]):
            print(f"{i+1}. Doc: {analysis['document']}, Page: {analysis['page_number']}")
            print(f"    Refined Text: {analysis['refined_text'][:150]}...") # Show first 150 chars
    else:
        print("No subsection analysis generated.")