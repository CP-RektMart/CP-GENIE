import requests
from bs4 import BeautifulSoup
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from langchain.schema import Document
import dotenv
import sys

dotenv.load_dotenv()

def get_all_sitemap_links(index_url: str) -> list[str]:
    res = requests.get(index_url)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "xml")

    links = soup.find_all("loc")
    sitemap_urls = [
        loc.text.strip()
        for loc in links
        if not loc.text.strip().endswith("sitemap-root.xml")
    ]
    return sitemap_urls


def parse_sitemap(sitemap_url):
    response = requests.get(sitemap_url)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    urls = []
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    for url in root.findall('ns:url', namespace):
        loc = url.find('ns:loc', namespace).text
        urls.append(loc)
    return urls


def scrape_docs(url: str) -> Document:
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    content_div = soup.select_one("div#content")
    if not content_div:
        raise ValueError(f"No #content found in {url}")

    body = content_div.get_text(separator="\n", strip=True)
    h1_post = soup.select_one("h1.title-post")
    if h1_post:
        title = h1_post.text.strip()
    elif soup.title:
        title = soup.title.string.strip()
    else:
        title = "Untitled"

    metadata = {
        "source": url,
        "title": title,
    }

    return Document(page_content=body, metadata=metadata)

def scrape() -> list[Document]:
    index_url = os.getenv("SITEMAP_URL")
    sitemap_urls = get_all_sitemap_links(index_url)
    
    urls = []
    for sitemap_url in tqdm(sitemap_urls, position=0, leave=True):
        urls += parse_sitemap(sitemap_url)

    docs = []
    for url in tqdm(urls, desc="Scraping URLs", position=0, leave=True):
        try:
            doc = scrape_docs(url)
            docs.append(doc)
        except ValueError as e:
            print(f"[SKIP] {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error on {url}: {e}")
    
    return docs