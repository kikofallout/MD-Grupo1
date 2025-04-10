from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from modules.spaCy_utils import process_text
from modules.mongoDB_utils import save_to_mongo_and_pinecone
from tqdm import tqdm  # Adicionando a importação do tqdm
import requests
from requests.adapters import HTTPAdapter, Retry
import os
import requests
from requests.adapters import HTTPAdapter, Retry
import re
from datetime import datetime
import time
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import pdfplumber
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from io import BytesIO
import requests
import fitz  # PyMuPDF
from io import BytesIO
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from modules.mongoDB_utils import save_to_mongo_and_pinecone
from collections import Counter
import string
import spacy
from modules.spaCy_utils import process_text
from modules.spaCy_utils import normalize_text
import os
from docx import Document
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from collections import Counter
import urllib.request
import fitz
from modules.spaCy_utils import normalize_text
from modules.spaCy_utils import generate_embeddings
from modules.spaCy_utils import food_matcher, nutrient_matcher, diet_matcher, eating_habits_matcher

BASE_URL = "https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials"
#BASE_URL = "https://www.dietaryguidelines.gov"

def extract_text_from_pdf_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        # Abrir PDF a partir do conteúdo em bytes
        with BytesIO(response.content) as pdf_file:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for i, page in enumerate(doc):
                print(f"📄 Extraindo texto da página {i+1}/{len(doc)}")
                text += page.get_text()
            return text

    except Exception as e:
        print(f"❌ Erro ao extrair texto do PDF de {url}: {e}")
        return None
    
def get_pdf_links_selenium():
    """Usa Selenium para extrair os links dos PDFs da subpágina."""
    print("🚀 Iniciando o Selenium...")
    chrome_options = Options()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")  # Desabilita GPU no headless mode
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # adapta se necessário

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        print("🔎 Acessando a página com Selenium...")
        print("🌐 URL:", BASE_URL)

        driver.get(BASE_URL)

        # Espera até que pelo menos um link de PDF esteja presente
        WebDriverWait(driver, 60).until(
            #EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '.pdf')]"))
            #EC.presence_of_element_located((By.TAG_NAME, "a")) 
            EC.presence_of_element_located((By.ID, "main-content"))
        )

        time.sleep(5) 
        print("✅ Página carregada com links visíveis.")

        html = driver.page_source
    except Exception as e:
        print(f"❌ Erro ao carregar conteúdo da página: {e}")
        driver.quit()
        return []
    
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []

    for a in soup.find_all("a", href=True):
        #print("➡️", a['href'])
        href = a['href']
        if href.endswith(".pdf"):
            full_url = href if href.startswith("http") else f"https://www.dietaryguidelines.gov{href}"
            pdf_links.append(full_url)

    print(f"✅ {len(pdf_links)} links de PDF encontrados com Selenium.")
    return list(set(pdf_links))

def process_dietary_guidelines_pdfs():
    print("📥 Buscando links de PDFs no site via Selenium...")
    pdf_links = get_pdf_links_selenium()
    print(f"🔗 {len(pdf_links)} links encontrados.")

    return pdf_links

def process_dietary_pdfs():
    """Função principal para coletar e processar os PDFs da Dietary Guidelines"""
    print("📥 Buscando links de PDFs no site via Selenium...")
    pdf_links = process_dietary_guidelines_pdfs()

    if not pdf_links:
        print("❌ Nenhum link de PDF encontrado.")
        return

    # Passando os links encontrados para o processamento
    process_pdf_links(pdf_links)

def download_pdf(url, file_path):
    """Função para baixar o PDF usando urllib e salvar localmente."""

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path)) 

    try:
        #response = requests.get(url, stream=True)
        urllib.request.urlretrieve(url, file_path)
        print(f"✅ PDF baixado com sucesso: {file_path}")

    except Exception as e:
        print(f"❌ Erro ao baixar o PDF: {e}")
        return None
    return file_path

def safe_nlp_processing(text, nlp, max_tokens=512):
    """Processa texto em blocos menores para evitar ultrapassar o limite de tokens do modelo."""
    from spacy.tokens import DocBin
    chunks = []
    current_text = ""
    current_tokens = 0

    # Dividimos o texto em parágrafos
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue

        token_count = len(paragraph.split())
        if current_tokens + token_count <= max_tokens:
            current_text += paragraph + "\n"
            current_tokens += token_count
        else:
            doc = nlp(current_text.strip())
            chunks.append(doc)
            current_text = paragraph + "\n"
            current_tokens = token_count

    if current_text.strip():
        doc = nlp(current_text.strip())
        chunks.append(doc)

    return chunks


def process_pdf_links(pdf_urls):
    """Processa PDFs ou DOCXs da Dietary Guidelines e extrai texto, metadados e vetores."""
    articles = []

    # Certifique-se de que a pasta de download existe
    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    print(f"📄 A processar {len(pdf_urls)} ficheiros...")
    for url in tqdm(pdf_urls, desc="📥 A fazer download e extração"):
        time.sleep(10)
        try:
            print(f"🚀 Tentando baixar: {url}")
            
            text = extract_text_from_pdf_url(url)
            
            if not text or not text.strip():
                print(f"⚠️ Nenhum texto extraído: {url}")
                continue

            print(f"🚀 Tentando baixar3: {url}")
            
            file_name = url.split("/")[-1].split("?")[0]
            file_extension = file_name.split(".")[-1]
            title = file_name.replace("-", " ").replace(f".{file_extension}", "").title()

            # Processamento do texto com spaCy
            #spacy_results = process_text(text)

            # Normaliza texto antes de dividir
            normalized_text = normalize_text(text)
            spacy_chunks = safe_nlp_processing(normalized_text, nlp)

            entities = []
            categorized_entities = []
            matched_terms = {
                "FOOD": [],
                "NUTRIENT": [],
                "DIET": [],
                "EATING_HABITS": []
            }
            chunks = []
            embeddings = []

            for doc in spacy_chunks:
                ents = [(ent.text, ent.label_) for ent in doc.ents]
                entities.extend(ents)

                # Aplica os matchers em cada chunk
            matches = {
                "FOOD": [doc[start:end].text for _, start, end in food_matcher(doc)],
                "NUTRIENT": [doc[start:end].text for _, start, end in nutrient_matcher(doc)],
                "DIET": [doc[start:end].text for _, start, end in diet_matcher(doc)],
                "EATING_HABITS": [doc[start:end].text for _, start, end in eating_habits_matcher(doc)]
            }

            # Acumula matches globais
            for category, terms in matches.items():
                matched_terms[category].extend(terms)

            # Classifica entidades que batem com termos de interesse
            for ent_text, _ in ents:
                for category, terms in matches.items():
                    if ent_text in terms:
                        categorized_entities.append((ent_text, category))
                        break

            chunks.append(doc.text)

            # Gera embeddings baseados nos chunks extraídos
            embeddings = generate_embeddings(chunks)

            spacy_results = {
                "entities": categorized_entities,
                "matched_terms": matched_terms,
                "chunks": chunks,
                "embeddings": embeddings
            }

            keywords = extract_keywords_from_text(text, top_n=8)
            abstract = summarize_text(text, nlp)

            # Armazenando os dados extraídos
            articles.append({
                "title": title,
                "text": text.strip(),
                "abstract": abstract,
                "authors": "",
                "keywords": keywords,
                "doi": "",
                "journal": "",
                "last_updated": "",
                "year": 2023,
                "spacy_entities": spacy_results["entities"],
                "spacy_matched_terms": spacy_results["matched_terms"],
                "chunks": spacy_results["chunks"],
                "embeddings": spacy_results["embeddings"],
                "source": "DietaryGuidelines"
            })

        except Exception as e:
            print(f"⚠️ Erro ao processar {url}: {e}")
            continue

    if not articles:
        print("❌ Nenhum ficheiro processado com sucesso.")
        return

    # Salvando os artigos extraídos no MongoDB e Pinecone
    save_to_mongo_and_pinecone(articles, "DietaryGuidelines")
    print("✅ Ficheiros salvos no MongoDB e Pinecone!")

# Carregar modelo spacy
#nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("en_core_web_sm")  

def extract_keywords_from_text(text, top_n=8):
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
    keywords = [word for word in words if word not in stopwords and len(word) > 3]
    most_common = Counter(keywords).most_common(top_n)
    return [kw for kw, _ in most_common]

def summarize_text(text, nlp, max_sentences=3):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
    return " ".join(sentences[:max_sentences])

def extract_text_from_docx(file_stream):
    """Extrai texto de um ficheiro .docx."""
    document = Document(file_stream)
    full_text = [para.text for para in document.paragraphs]
    return "\n".join(full_text)