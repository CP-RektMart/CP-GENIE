from langchain_google_genai import ChatGoogleGenerativeAI

import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import pytesseract
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001"
)


tesseract_lang = 'tha'

# URL to download PDF from
url = "https://www.cp.eng.chula.ac.th/downloadformcurrent"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

pdf_links = []

# Extract .pdf links
for a_tag in soup.find_all("a", href=True):
    href = a_tag["href"]
    if href.endswith(".pdf"):
        full_link = requests.compat.urljoin(url, href)
        pdf_links.append(full_link)

# Download the first PDF (for example)
pdf_url = pdf_links[2]
print(f"Downloading PDF from: {pdf_url}")
pdf_response = requests.get(pdf_url)
pdf_path = "temp.pdf"
with open(pdf_path, "wb") as f:
    f.write(pdf_response.content)

# Convert PDF to images
images = convert_from_path(pdf_path)

# OCR each image
text_output = ""
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang=tesseract_lang)
    text_output += f"\n--- Page {i+1} ---\n{text}"

# Save or print output
with open("output_thai_text.txt", "w", encoding="utf-8") as f:
    f.write(text_output)

print("✅ OCR completed. Output saved to output_thai_text.txt")