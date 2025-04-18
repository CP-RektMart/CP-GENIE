import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import re
import hashlib
import xml.etree.ElementTree as ET
from typing import List, Optional
import json
from datetime import datetime
import mimetypes

# --- Configuration ---
START_URL = "https://www.cp.eng.chula.ac.th/"
BASE_DOMAIN = urlparse(START_URL).netloc
CONTENT_SELECTOR = "div[role='main']"

# --- Updated Directory Structure ---
BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
HTML_DIR = os.path.join(RAW_DIR, "html")
PDF_DIR = os.path.join(RAW_DIR, "pdf")
IMAGES_DIR = os.path.join(RAW_DIR, "images")
OFFICE_DIR = os.path.join(RAW_DIR, "office")

PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TEXT_DIR = os.path.join(PROCESSED_DIR, "text")
METADATA_DIR = os.path.join(PROCESSED_DIR, "metadata")

# Create all required directories
for directory in [HTML_DIR, PDF_DIR, IMAGES_DIR, OFFICE_DIR, TEXT_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Politeness Delay ---
DELAY = 2

# --- Set to keep track of visited URLs ---
visited_urls = set()

# --- Queue for URLs to scrape ---
from collections import deque

# --- Content type mapping ---
CONTENT_TYPE_DIRS = {
    "text/html": HTML_DIR,
    "application/pdf": PDF_DIR,
    "application/msword": OFFICE_DIR,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": OFFICE_DIR,
    "application/vnd.ms-powerpoint": OFFICE_DIR,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": OFFICE_DIR,
    "application/vnd.ms-excel": OFFICE_DIR,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": OFFICE_DIR,
    "image/jpeg": IMAGES_DIR,
    "image/png": IMAGES_DIR,
    "image/gif": IMAGES_DIR,
    "image/webp": IMAGES_DIR,
}


def url_to_filename(url):
    """Convert URL to a filename by replacing slashes with double underscores and removing illegal chars."""
    # Parse the URL to extract the path and query
    parsed = urlparse(url)
    path = parsed.path

    # Remove leading and trailing slashes
    path = path.strip("/")

    # Replace slashes with double underscores
    filename = path.replace("/", "__")

    # Add query parameters if they exist
    if parsed.query:
        filename += f"__{parsed.query}"

    # Remove illegal characters
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)

    # If the filename is empty (e.g., for the root URL), use 'index'
    if not filename:
        filename = "index"

    # Add a unique hash to avoid collisions
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

    return f"{filename}__{url_hash}"


def get_content_type_dir(content_type):
    """Determine the appropriate directory based on content type."""
    # Get the base content type (ignore parameters)
    base_type = content_type.split(";")[0].strip()

    # Check if it's a known content type
    if base_type in CONTENT_TYPE_DIRS:
        return CONTENT_TYPE_DIRS[base_type]

    # For other image types
    if base_type.startswith("image/"):
        return IMAGES_DIR

    # For other office document types
    if "officedocument" in base_type or "ms-" in base_type:
        return OFFICE_DIR

    # Default to HTML for unknown types
    return HTML_DIR


def create_metadata(url, content_type, filepath):
    """Create metadata JSON for a file."""
    metadata = {
        "source_url": url,
        "content_type": content_type.split(";")[0].strip(),
        "fetch_timestamp": datetime.now().isoformat(),
        "raw_filepath": filepath,
    }

    # Determine the base filename (without extension)
    filename = os.path.basename(filepath)
    base_filename = os.path.splitext(filename)[0]

    # Create metadata file path
    metadata_path = os.path.join(METADATA_DIR, f"{base_filename}.json")

    # Write metadata to file
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def extract_text_content(content, content_type):
    """Extract text content from various types of content."""
    # For HTML content
    if content_type.startswith("text/html"):
        soup = BeautifulSoup(content, "lxml")
        # Extract text from the main content area
        content_area = soup.select_one(CONTENT_SELECTOR)
        if content_area:
            return content_area.get_text(separator="\n", strip=True)
        return ""
    # # For PDF content
    # elif content_type == "application/pdf":
    #     # PDF text extraction would require additional libraries (e.g., PyPDF2)
    #     # Placeholder for PDF extraction
    #     return "PDF content extraction not implemented."
    # # For image content
    # elif content_type.startswith("image/"):
    #     # Image text extraction would require OCR (e.g., pytesseract)
    #     # Placeholder for image extraction
    #     return "Image content extraction not implemented."
    # # For office documents
    # elif "officedocument" in content_type or "ms-" in content_type:
    #     # Office document text extraction would require additional libraries (e.g., python-docx)
    #     # Placeholder for office document extraction
    #     return "Office document content extraction not implemented."
    # # For JSON content
    # elif content_type == "application/json":
    #     try:
    #         json_content = json.loads(content)
    #         return json.dumps(json_content, indent=2, ensure_ascii=False)
    #     except json.JSONDecodeError:
    #         return "Invalid JSON content."
    # # For text files
    # elif content_type.startswith("text/"):
    #     # Assuming text files are UTF-8 encoded
    #     return content.decode("utf-8", errors="ignore")
    # # For XML content
    # elif content_type.startswith("application/xml") or content_type.startswith(
    #     "text/xml"
    # ):
    #     try:
    #         xml_content = ET.fromstring(content)
    #         return ET.tostring(xml_content, encoding="unicode")
    #     except ET.ParseError:
    #         return "Invalid XML content."

    # For other types, return empty string (text extraction would require additional libraries)
    return ""


def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Extract URLs from a sitemap.xml file"""
    try:
        response = requests.get(sitemap_url, timeout=15)
        response.raise_for_status()

        # Handle both direct sitemap and sitemap index files
        root = ET.fromstring(response.content)

        # Namespace handling
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        urls = []

        # Check if this is a sitemap index
        sitemaps = root.findall(".//sm:sitemap/sm:loc", ns)
        if sitemaps:
            # This is a sitemap index, recursively process each sitemap
            for sitemap in sitemaps:
                child_urls = get_urls_from_sitemap(sitemap.text)
                urls.extend(child_urls)
        else:
            # This is a regular sitemap
            for url in root.findall(".//sm:url/sm:loc", ns):
                urls.append(url.text)

        return urls

    except Exception as e:
        print(f"Error processing sitemap {sitemap_url}: {e}")
        return []


# Test URLs
test_urls = [
    "https://www.cp.eng.chula.ac.th/blog/archives/35393",
    # "https://www.cp.eng.chula.ac.th/blog/archives/48",
    # "https://www.cp.eng.chula.ac.th/blog/archives/34948",
    # "https://www.cp.eng.chula.ac.th/future/bachelor2018",
]

queue = deque(test_urls)

# --- Main Scraping Loop ---
while queue:
    current_url = queue.popleft()

    if current_url in visited_urls:
        continue

    print(f"Scraping: {current_url}")
    visited_urls.add(current_url)

    try:
        # --- Make the HTTP request ---
        headers = {"User-Agent": "Scraper Bot (Academic Project; respectful scraping)"}
        response = requests.get(current_url, headers=headers, timeout=15)
        response.raise_for_status()

        # --- Get content type ---
        content_type = response.headers.get("Content-Type", "text/html")

        # --- Determine file extension based on content type ---
        extension = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if not extension:
            # Default to .html if extension can't be determined
            extension = ".html"

        # --- Create filename based on URL ---
        base_filename = url_to_filename(current_url)
        filename = f"{base_filename}{extension}"

        # --- Determine which directory to save in based on content type ---
        save_dir = get_content_type_dir(content_type)
        filepath = os.path.join(save_dir, filename)

        # --- Save raw content ---
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"  Saved raw content to: {filepath}")

        # --- Create metadata ---
        metadata_path = create_metadata(current_url, content_type, filepath)
        print(f"  Created metadata at: {metadata_path}")

        # --- Extract and save text if possible ---
        text_content = extract_text_content(response.content, content_type)
        if text_content:
            text_filepath = os.path.join(TEXT_DIR, f"{base_filename}.txt")
            with open(text_filepath, "w", encoding="utf-8") as f:
                # f.write(f"Source URL: {current_url}\n\n")
                f.write(text_content)
            print(f"  Saved text content to: {text_filepath}")

        # --- Extract links if it's HTML content ---
        if content_type.startswith("text/html"):
            soup = BeautifulSoup(response.content, "lxml")
            content_area = soup.select_one(CONTENT_SELECTOR)

            if content_area:
                # --- Find and add new valid links to the queue ---
                for link in content_area.find_all("a", href=True):
                    href = link["href"]
                    absolute_url = urljoin(current_url, href)
                    parsed_url = urlparse(absolute_url)

                    if (
                        parsed_url.scheme in ["http", "https"]
                        and parsed_url.netloc == BASE_DOMAIN
                        and absolute_url not in visited_urls
                        and "#" not in absolute_url
                    ):

                        if absolute_url not in queue:
                            queue.append(absolute_url)

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {current_url}: {e}")
    except Exception as e:
        print(f"  An unexpected error occurred for {current_url}: {e}")

    # --- Politeness delay ---
    time.sleep(DELAY)
    break  # Uncomment this line to continue scraping all URLs in the queue

# Debugging output
print("-" * 20)
print("Visited URLs:")
for url in visited_urls:
    print(url)

print("-" * 20)
print("Remaining URLs in queue:")
for url in queue:
    print(url)

print("-" * 20)
print(f"Scraping finished. Visited {len(visited_urls)} pages.")
