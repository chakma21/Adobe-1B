import json
import datetime
from typing import List, Dict, Any
import os

# --- Constants for file paths ---
# These are defined globally for easy access throughout the script
input_json_filename = "input_payload.json"
documents_directory = "documents" # PDFs are expected to be in this directory
output_filename = "processed_document_output.json"
intermediate_candidate_sections_filename = "intermediate_candidate_sections.json" # New intermediate file

# --- Placeholder for PDF Text Extraction ---
# You'll need to install a library like PyPDF2
# pip install PyPDF2
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page by page from a PDF and attempts to identify sections.
    This implementation uses a simple heuristic for sectioning based on lines and empty lines.
    For more complex PDFs, a more sophisticated layout analysis method might be needed.
    """
    extracted_data = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue # Skip empty pages

            lines = text.split('\n')
            section_counter = 0
            current_section_text = ""
            for line in lines:
                current_section_text += line + "\n"
                # Simple sectioning logic: after 10 lines or a significant break
                if len(current_section_text.split('\n')) > 10 or (line.strip() == "" and len(current_section_text.strip()) > 50):
                    if current_section_text.strip():
                        extracted_data.append({
                            "document": os.path.basename(pdf_path),
                            "page_number": i + 1,
                            "section_title": f"Section {section_counter + 1} on Page {i + 1}", # Generic title
                            "full_text": current_section_text.strip()
                        })
                        section_counter += 1
                    current_section_text = ""
            # Add any remaining text as a section
            if current_section_text.strip():
                extracted_data.append({
                    "document": os.path.basename(pdf_path),
                    "page_number": i + 1,
                    "section_title": f"Section {section_counter + 1} on Page {i + 1}",
                    "full_text": current_section_text.strip()
                })

    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return extracted_data

# --- Placeholder for Sentence Embedding Model and Similarity ---
# Ensure 'sentence-transformers' and 'scikit-learn' are installed.
# The model ('all-MiniLM-L6-v2') must be pre-downloaded for offline execution.
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    # Attempt to load the model from a local path
    MODEL_PATH = './models/all-MiniLM-L6-v2'
    # Current date: July 22, 2025. This path should exist.
    embedding_model = SentenceTransformer(MODEL_PATH)
    print(f"Successfully loaded embedding model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load local embedding model at {MODEL_PATH}: {e}.")
    print("Proceeding with a dummy model. Please ensure 'all-MiniLM-L6-v2' is downloaded and in './models/'.")
    class DummyEmbeddingModel:
        def encode(self, texts):
            return np.random.rand(len(texts), 384) # Standard dimension for MiniLM models
    embedding_model = DummyEmbeddingModel()

def get_embedding(text: str):
    """Generates an embedding for a given text using the loaded model."""
    if isinstance(embedding_model, DummyEmbeddingModel):
        return embedding_model.encode([text])[0]
    return embedding_model.encode(text)

def calculate_relevance(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two text embeddings."""
    if not text1 or not text2:
        return 0.0
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

# --- Placeholder for Refined Text Extraction/Summarization ---
def refine_text_for_job(full_text: str, persona: str, job_to_be_done: str) -> str:
    """
    Refines a full text section to extract the most relevant parts based on persona and job.
    This is a basic keyword-based extraction; a more advanced solution would use NLP summarization.
    """
    # Create a simple query combining persona and job for keyword extraction
    query_keywords = (persona + " " + job_to_be_done).lower().replace(",", "").replace(".", "").split()
    relevant_sentences = []
    sentences = full_text.split('. ') # Split by sentence for refinement

    for sentence in sentences:
        # Check if any query keyword is present in the sentence
        if any(keyword in sentence.lower() for keyword in query_keywords if len(keyword) > 2): # Ignore very short words
            relevant_sentences.append(sentence.strip())
        if len(relevant_sentences) >= 3: # Limit the number of extracted sentences for brevity
            break

    if relevant_sentences:
        return ". ".join(relevant_sentences).strip() + "."
    
    # Fallback: if no specific relevant sentences, return the first sentence or an empty string
    return full_text.split('. ')[0].strip() + "..." if full_text else ""


def process_documents(input_data: Dict[str, Any], docs_dir: str) -> Dict[str, Any]:
    """
    Processes the input documents, calculates relevance, ranks sections,
    and generates the final structured output.
    """
    metadata = {
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "persona": input_data["persona"]["role"],
        "job_to_be_done": input_data["job_to_be_done"]["task"],
        "processing_timestamp": datetime.datetime.now().isoformat()
    }

    all_candidate_sections = [] # List to store all extracted sections before ranking

    job_description_query = f"Persona: {input_data['persona']['role']}. Job: {input_data['job_to_be_done']['task']}"

    print("\n--- Starting document processing ---")
    for doc_info in input_data["documents"]:
        pdf_filename = doc_info["filename"]
        pdf_path = os.path.join(docs_dir, pdf_filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: Document not found at '{pdf_path}'. Skipping this document.")
            continue

        print(f"Extracting text from: {pdf_path}")
        extracted_sections_from_pdf = extract_text_from_pdf(pdf_path)

        for section_data in extracted_sections_from_pdf:
            section_text = section_data["full_text"]
            relevance_score = calculate_relevance(job_description_query, section_text)
            
            all_candidate_sections.append({
                "document": section_data["document"],
                "page_number": section_data["page_number"],
                "section_title": section_data["section_title"],
                "full_text": section_text,
                "importance_rank_score": relevance_score
            })
    print("--- Document processing complete ---")

    # Save all candidate sections to the intermediate file
    with open(intermediate_candidate_sections_filename, 'w', encoding='utf-8') as f:
        json.dump(all_candidate_sections, f, indent=4)
    print(f"Intermediate candidate sections saved to '{intermediate_candidate_sections_filename}'")

    # Sort all candidate sections by importance score in descending order for final ranking
    all_candidate_sections.sort(key=lambda x: x["importance_rank_score"], reverse=True)

    # Prepare final output structures based on the sorted sections
    output_extracted_sections = []
    output_subsection_analysis = []

    # Define how many top sections to include in the 'extracted_sections' part
    num_top_sections_for_output = min(5, len(all_candidate_sections)) 

    # Populate 'extracted_sections' with the very top-ranked sections
    for i in range(num_top_sections_for_output):
        section = all_candidate_sections[i]
        output_extracted_sections.append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": i + 1, # Assign rank based on sorted order
            "page_number": section["page_number"]
        })

    # Populate 'subsection_analysis' with refined text for relevant sections.
    # We can use a relevance threshold to include more than just the top N sections.
    relevance_threshold_for_analysis = 0.5 # Sections with score >= 0.5 are considered for detailed analysis

    # Filter and sort sections for subsection analysis. Sorting by doc/page for consistent output order.
    filtered_for_analysis = sorted(
        [s for s in all_candidate_sections if s["importance_rank_score"] >= relevance_threshold_for_analysis],
        key=lambda x: (x["document"], x["page_number"])
    )

    seen_subsection_entries = set() # Helps prevent duplicate entries if refined_text is identical

    for section in filtered_for_analysis:
        refined_text = refine_text_for_job(
            section["full_text"], 
            metadata["persona"], 
            metadata["job_to_be_done"]
        )
        if refined_text:
            # Use a tuple for the set to uniquely identify entries
            entry_key = (section["document"], section["page_number"], refined_text) 
            if entry_key not in seen_subsection_entries:
                output_subsection_analysis.append({
                    "document": section["document"],
                    "refined_text": refined_text,
                    "page_number": section["page_number"]
                })
                seen_subsection_entries.add(entry_key)

    return {
        "metadata": metadata,
        "extracted_sections": output_extracted_sections,
        "subsection_analysis": output_subsection_analysis
    }

# --- Main execution block ---
if __name__ == "__main__":
    # --- Step 1: Load Input Payload ---
    if not os.path.exists(input_json_filename):
        print(f"Error: Input JSON file '{input_json_filename}' not found.")
        print("Please ensure '{input_json_filename}' is in the same directory as this script.")
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

    # job_to_be_done_query is not directly used here but was from your provided block
    # job_to_be_done_query = input_payload["job_to_be_done"]["task"]

    # --- Step 2: Ensure Documents Directory Exists ---
    os.makedirs(documents_directory, exist_ok=True)
    print(f"Ensured '{documents_directory}' directory exists.")

    # --- Step 3: Create Dummy PDF files for Demonstration (if they don't exist) ---
    # This block helps in setting up the environment for testing.
    # In a real deployment, you'd ensure actual PDFs are placed in the 'documents' folder.
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        dummy_pdf_names = [
            "South of France - Cities.pdf",
            "South of France - Cuisine.pdf",
            "South of France - History.pdf",
            "South of France - Restaurants and Hotels.pdf",
            "South of France - Things to Do.pdf",
            "South of France - Tips and Tricks.pdf",
            "South of France - Traditions and Culture.pdf"
        ]

        for pdf_name in dummy_pdf_names:
            pdf_path = os.path.join(documents_directory, pdf_name)
            if not os.path.exists(pdf_path):
                c = canvas.Canvas(pdf_path, pagesize=letter)
                c.drawString(100, 750, f"This is a dummy page 1 for {pdf_name}.")
                c.drawString(100, 730, "It contains some sample text to simulate a real PDF.")
                c.drawString(100, 710, "This section is about general information.")
                
                # Adding some specific content to make dummy PDFs more relevant for testing
                if "Cities" in pdf_name:
                    c.drawString(100, 600, "Comprehensive Guide to Major Cities in the South of France")
                    c.drawString(100, 580, "Nice offers beautiful beaches and the Promenade des Anglais, perfect for relaxation.")
                    c.drawString(100, 560, "Marseille is a vibrant port city with rich history and many cultural sites.")
                    c.showPage()
                    c.drawString(100, 750, f"Page 2 of {pdf_name}. More city details.")
                    c.drawString(100, 730, "Cannes is famous for its film festival and luxury resorts, ideal for entertainment.")
                elif "Things to Do" in pdf_name:
                    c.drawString(100, 600, "Coastal Adventures")
                    c.drawString(100, 580, "The South of France boasts a beautiful Mediterranean coastline. Enjoy beach hopping in Nice or exploring the Calanques near Marseille. Water sports like jet skiing and snorkeling are popular.")
                    c.drawString(100, 500, "Nightlife and Entertainment")
                    c.drawString(100, 480, "Experience the vibrant nightlife in Saint-Tropez or Nice, with options ranging from chic bars to lively nightclubs. Look for live music and DJ events.")
                    c.showPage()
                    c.drawString(100, 750, f"Page 2 of {pdf_name} - More Activities.")
                    c.drawString(100, 730, "Consider boat tours along the coast or exploring the charming villages inland.")
                elif "Cuisine" in pdf_name:
                    c.drawString(100, 600, "Culinary Experiences")
                    c.drawString(100, 580, "Taste local delicacies like bouillabaisse and ratatouille. Many towns offer cooking classes and wine tours in Provence and Languedoc.")
                    c.showPage()
                    c.drawString(100, 750, f"Page 2 of {pdf_name} - Dining Tips.")
                    c.drawString(100, 730, "Explore local markets for fresh produce and regional specialties. Dining at beachfront restaurants is a must.")
                elif "Tips and Tricks" in pdf_name:
                    c.drawString(100, 600, "General Packing Tips and Tricks")
                    c.drawString(100, 580, "Pack light and in layers, as weather can vary. Don't forget travel-sized toiletries, reusable bags, and copies of important documents.")

                c.save()
                print(f"Created dummy PDF: '{pdf_path}'")
    except ImportError:
        print(f"Warning: 'reportlab' not installed. Cannot create dummy PDFs. Please ensure actual PDFs exist in '{documents_directory}' for testing.")
        print("To install 'reportlab': pip install reportlab")

    # --- Step 4: Process Documents and Generate Output ---
    print("\nStarting the document analysis process...")
    output_result = process_documents(input_payload, documents_directory)

    # --- Step 5: Save Final Output ---
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_result, f, indent=4)
        print(f"Final processed output saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving final output to '{output_filename}': {e}")

    print("\nDocument analysis complete. Check the output files.")