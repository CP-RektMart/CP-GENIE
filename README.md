# CP-Genie

A RAG-based chatbot system built with LangGraph, LangChain, and FastAPI.  

---

## Installation

Make sure you have **Python 3.11+** and [**Poetry**](https://python-poetry.org/), Tesseract 5.5.0 installed.

```bash
# Clone the repo
git clone https://github.com/CP-RektMart/CP-GENIE.git
cd cp-genie

# Install dependencies using Poetry
poetry install

# Activate virtual env
poetry env activate

# Start the FastAPI
poetry run start

```

OCR 

```bash
# Install tesseract
https://github.com/UB-Mannheim/tesseract/wiki and add C:\Program Files\Tesseract-OCR to path env (window) or sudo apt install tesseract-ocr

# Install poppler(idk in case you use linux this is window version)
https://github.com/oschwartz10612/poppler-windows/releases & add C:\poppler-24.08.0\Library\bin to path env

# Install tha.traineddata and put into C:\Program Files\Tesseract-OCR\tessdata\ (again I dont know in case you use linux but this work on window)
https://github.com/tesseract-ocr/tessdata/raw/main/tha.traineddata
```
