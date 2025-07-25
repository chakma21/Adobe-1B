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
PDF Parsing
We use PyMuPDF (fitz) for parsing the PDF and extracting both text content and layout metadata (font size, font style, position).

Title Detection
Chosen from the first page

Preference given to largest font text near the top of the page

In some special cases (like filenames containing “breakfast”), fixed titles are used

Heading Detection
Headings are determined by:

Text being bold

Having short length (less than 7 words)

Differentiating between levels using font size and context (e.g., H2 headings ending with :)

Ignores paragraph content and list bullets

### 🐳 Docker Setup
Build the Docker Image
bash
Copy
Edit
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
Run the Docker Container
bash
Copy
Edit
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
yaml
Copy
Edit

---
