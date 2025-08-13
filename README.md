# E-Library (Streamlit + Python)

A full-stack E‑Library built with Streamlit, Python, PyMuPDF for PDF processing, FAISS + SentenceTransformers for semantic search, and pytesseract for OCR student ID verification.

## Features
- Login with username/password and Student ID card OCR verification
- Personalized dashboard with suggested books
- Semantic search across PDFs (books and past year papers)
- References extraction from PDFs
- Previous year papers: filter by subject and year, search and download

## Project Structure
```
app/
  app.py
  backend/
    __init__.py
    auth.py
    pdf_utils.py
    search_index.py
  data/
    students.json
    books/
    papers/
  storage/
```

## Prerequisites
- Python 3.10+
- System dependency: Tesseract OCR
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
streamlit run app/app.py
```

## Data
- Place book PDFs under `app/data/books`
- Place past year papers under `app/data/papers/<Subject>/` and include the year in the filename, e.g., `app/data/papers/Mathematics/Mathematics_2022.pdf`

## Notes
- The first run will download the `all-MiniLM-L6-v2` embedding model.
- OCR verification requires Tesseract to be installed on the system.
