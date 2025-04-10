import requests
import fitz  # PyMuPDF
from io import BytesIO
from tqdm import tqdm
from modules.mongoDB_utils import save_to_mongo_and_pinecone
from collections import Counter
import string
import spacy
from modules.spaCy_utils import process_text
from modules.spaCy_utils import normalize_text
import os
from docx import Document  
from collections import Counter
import string
import spacy
nlp = spacy.load("en_core_web_trf")


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

def process_pdf_links(pdf_urls):
    """Processa PDFs ou DOCXs da EatRight e extrai texto, metadados e vetores."""
    articles = []

    print(f"📄 A processar {len(pdf_urls)} ficheiros...")
    for url in tqdm(pdf_urls, desc="📥 A fazer download e extração"):
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"❌ Falha ao fazer download: {url}")
                continue

            file_extension = os.path.splitext(url.split("?")[0])[1].lower()
            file_stream = BytesIO(response.content)

            if file_extension == ".pdf":
                doc = fitz.open(stream=file_stream, filetype="pdf")
                text = "".join(page.get_text() for page in doc)

            elif file_extension == ".docx":
                text = extract_text_from_docx(file_stream)

            else:
                print(f"⚠️ Formato não suportado ({file_extension}): {url}")
                continue

            if not text.strip():
                print(f"⚠️ Ficheiro sem texto extraível: {url}")
                continue

            title = url.split("/")[-1].split("?")[0].replace("-", " ").replace(file_extension, "").title()
            spacy_results = process_text(text)
            keywords = extract_keywords_from_text(text, top_n=8)
            abstract = summarize_text(text, nlp)

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
                "source": "EatRight"
            })

        except Exception as e:
            print(f"⚠️ Erro ao processar {url}: {e}")
            continue

    if not articles:
        print("❌ Nenhum ficheiro processado com sucesso.")
        return

    save_to_mongo_and_pinecone(articles, "EatRight")
    print("✅ Ficheiros salvos no MongoDB e Pinecone!")
