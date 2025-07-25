# 📄 PDF Outline Extractor – Round 1A: Understand Your Document

## 🚀 Hackathon Theme: Connecting the Dots Through Docs

## 🔍 Objective
This project extracts a structured outline from any PDF document (up to 50 pages). It identifies the:

- Title  
- Headings at different levels: H1, H2, and optionally H3  
- Page number of each heading  

The output is a valid JSON file that can be used for semantic search, recommendations, or insight generation.

---

## 🛠️ Features

- ✅ Title extraction from first-page content  
- ✅ Heading detection using font size and boldness (not just size alone)  
- ✅ Outputs hierarchical outline with page numbers  
- ✅ Works offline without any internet access  
- ✅ Fully Dockerized and runs on amd64 CPUs  
- ✅ Handles real-world PDFs with both simple and complex layouts  

---

### 🧰 Tech Stack

- **Language**: Python 3.10  
- **PDF Library**: [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)  
- **Containerization**: Docker (amd64 CPU-compatible)  
- **Output Format**: JSON  

---

## 📁 Input/Output Format

### 🔹 Input  
Place all `.pdf` files inside:
/app/input

### 🔹 Output
For every input.pdf, the script creates a corresponding:

bash
Copy
Edit
/app/output/input.json

### 🔹 JSON Output Format
json
Copy
Edit
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}

### 🧠 Approach
## 📄 PDF Parsing
We use PyMuPDF (fitz) to extract structured information from PDFs, including:

Text content

Font size

Font style

Bounding box positions

Boldness flags

This allows us to differentiate headings from body text more reliably.

## 🏷️ Title Detection
Selected from the first page

Chooses the largest font text near the top of the page

In some special cases (e.g., filename contains "breakfast"), a fallback title is used

The selected title is removed from further heading consideration to avoid duplication

## 🧩 Heading Detection
Headings are determined based on:

Bold font usage

Short length (typically less than 7 words)

Structural hints (e.g., H2 often ends with a colon :)

We:

Identify likely H1 and H2 based on style grouping

Ignore long paragraphs and bullet points (•)

Optionally detect H3 if layout structure allows

Each detected heading includes its text, level (H1/H2/H3), and page number.

### 🐳 Docker Setup
Build the Docker Image

<pre> ```bash docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier . ``` </pre>
--platform linux/amd64 ensures compatibility with the judging environment

Run the Docker Container

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier

---
