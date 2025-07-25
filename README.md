### 🧠 Persona-Driven Document Intelligence – Round 1B

## 🚀 Hackathon Theme: Connecting the Dots Through Docs

## 🔍 Objective

This project extracts the most relevant sections from a set of PDF documents by understanding a given persona and their job-to-be-done. It ranks the extracted sections based on relevance and generates a summarized JSON output.

Works seamlessly across research papers, financial reports, whitepapers, and more.

---

## 🛠️ Features

- ✅ Extracts text from PDFs with structure (font size, position, boldness)
- ✅ Ranks sections based on persona & task using SentenceTransformers
- ✅ Summarizes the most relevant passages
- ✅ Outputs final results in a clean JSON format
- ✅ Dockerized and works offline (no internet required)
- ✅ CPU-only & <1GB model footprint

---

### 🧰 Tech Stack

- **Language**: Python 3.10
- **PDF Library**: PyMuPDF (fitz)
- **ML Models**: SentenceTransformers (CPU only)
- **Containerization**: Docker (amd64 CPU-compatible)
- **Output Format**: JSON

---

## 📁 Input/Output Format

### 🔹 Input

- Place all .pdf files inside:/app/documents
- Include an input_payload.json file with this format:

``` json
{
  "persona": {
    "role": "Investment Analyst"
  },
  "job_to_be_done": {
    "task": "Analyze revenue trends, R&D investments, and market positioning"
  },
  "documents": [
    {
      "filename": "CompanyA.pdf",
      "title": "Company A"
    },
    {
      "filename": "CompanyB.pdf"
    },
    {
      "filename": "CompanyC.pdf"
    }
  ]
}
```

### 🔹 Output

- The tool generates:/app/processed_document_output.json

```json
{
  "metadata": {
    "persona": "Investment Analyst",
    "job_to_be_done": "Analyze revenue trends...",
    "processing_timestamp": "2025-07-25T12:34:56"
  },
  "extracted_sections": [
    {
      "document": "CompanyA.pdf",
      "section_title": "Financial Overview",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "CompanyA.pdf",
      "refined_text": "Revenue increased by 15%...",
      "page_number": 3
    }
  ]
}
```
---

### 🧠 Approach

## 📄 PDF Parsing

- Extracts structured text blocks using PyMuPDF
- Captures font size, boldness, position, and style

---

## 🧩 Section Detection

- Uses font styles to identify headings (H1, H2, etc.)
- Groups content into sections for semantic ranking

---

## 🧠 Relevance Ranking

- Encodes persona/task and section text with SentenceTransformer
- Ranks sections using cosine similarity
- Selects top-k relevant sections

---

### 🐳 Docker Setup

Building the Docker Image


<pre> docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier . </pre>
For eg:-  <pre> ```docker build --platform linux/amd64 -t solution-1b:1.0 . ``` </pre>

--platform linux/amd64 ensures compatibility with the judging environment

Run the Docker Container
The command which runs the container in an isolated, offline environment

**This is very required step**
To get the exact path for the source
From the root directory, go to models folder then all-MiniLM-L6-v2
Then type pwd to get the path.
Then paste it in the docker run command.


For eg:- <pre> ```docker run --rm --network none --mount type=bind,source="$(pwd)",destination=/app --mount type=bind,source="C:\Users\shiny\Adobe\models\all-MiniLM-L6-v2",destination=/model solution-1b:1.0 ``` </pre>

#### Dont direct copy paste this command
###### Please refer to the pdf link to refer to the changes to be made in the command
