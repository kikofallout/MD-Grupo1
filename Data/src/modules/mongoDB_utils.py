import os
import uuid
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
from modules.spaCy_utils import process_text
from pinecone import Pinecone, ServerlessSpec
import json

def configure_mongoDB_connection():
    """Configure MongoDB connection."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["nutrition"] #md_nutrition_db
    collection = db["data"] #datas
    return collection

def configure_pinecone_connection():
    """Configure Pinecone connection."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "data"  
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Match BGE-small-en
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

def generate_unique_id():
    """Generate a unique ID for each paper."""
    return str(uuid.uuid4())

def extract_paper_attributes(paper, source):
    """Extract paper attributes based on the source API."""
    if source == "PubMed":
        year = paper.get("year", 0)
        if year == "No Year Available":
            year = 0
        return {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": int(year),
            "source": "PubMed",
            "abstract": paper.get("abstract", ""),
            "keywords": paper.get("keywords", []),
            "doi": paper.get("doi", ""),
            "journal": paper.get("journal", ""),
            "last_updated": paper.get("last_updated", "")
        }
    elif source == "Europe PMC":
        authors = paper.get("authorList", {}).get("author", [])
        authors = [f"{author.get('firstName', '')} {author.get('lastName', '')}" for author in authors]
        return {
            "title": paper.get("title", ""),
            "authors": authors,
            "year": int(paper.get("pubYear", 0) or 0),
            "source": "Europe PMC",
            "abstract": paper.get("abstractText", ""),
            "keywords": paper.get("keywordList", {}).get("keyword", []),
            "doi": paper.get("doi", ""),
            "journal": "",
            "last_updated": paper.get("firstPublicationDate", "")
        }
    elif source == "Semantic Scholar":
        return {
            "title": paper.get("title", ""),
            "authors": [author.get("name", "") for author in paper.get("authors", [])],
            "year": int(paper.get("year", 0) or 0),
            "source": "Semantic Scholar",
            "abstract": paper.get("abstract", ""),
            "keywords": [],
            "doi": paper.get("externalIds", {}).get("DOI", ""),
            "journal": paper.get("journal", {}).get("name", "") if paper.get("journal") else "",
            "last_updated": ""
        }
    elif source == "GoogleScholar":
        authors = paper.get("authors", "No Authors")
        if isinstance(authors, list):
            authors = ", ".join(authors)  # Combine authors into a string
        return {
            "title": paper.get("title", ""),
            "authors": authors,
            "year": int(paper.get("year", 0) or 0),
            "source": "Google Scholar",
            "abstract": paper.get("abstract", ""),
            "keywords": paper.get("keywords", []),
            "doi": paper.get("doi", ""),
            "journal": paper.get("journal", ""),
            "last_updated": ""
        }
    elif source == "EatRight":
        return {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year", 2023),
            "source": "EatRight",
            "abstract": paper.get("abstract", ""),
            "keywords": paper.get("keywords", []),
            "doi": "",
            "journal": "",
            "last_updated": paper.get("last_updated", "")
        }
    elif source == "DietaryGuidelines":
        return {
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year", 2025),
            "source": "Dietary Guidelines",
            "abstract": paper.get("abstract", ""),
            "keywords": paper.get("keywords", []),
            "doi": paper.get("doi", ""),
            "journal": paper.get("journal", ""),
            "last_updated": paper.get("last_updated", "")
        }
    elif source == "Wikipedia":
        return {
            "title": paper.get("title", ""),
            "authors": "Wikipedia Contributors",
            "year": 0,
            "source": "Wikipedia",
            "abstract": paper.get("summary", ""),
            "keywords": [],
            "doi": "",
            "journal": "Wikipedia",
            "last_updated": paper.get("url", "")  
        }


    else:
        raise ValueError(f"Unsupported source: {source}")


def save_paper_to_mongo_and_pinecone(paper, source, collection, index):
    """Save a single paper to MongoDB and Pinecone."""

    paper_data = extract_paper_attributes(paper, source)
    abstract = paper_data["abstract"]

    spacy_results = {
        "entities": paper.get("spacy_entities", []),
        "matched_terms": paper.get("spacy_matched_terms", {}),
        "chunks": paper.get("chunks", []),
        "embeddings": paper.get("embeddings", [])
    }

    if not spacy_results["chunks"] or spacy_results["embeddings"].shape[0] == 0:
        spacy_results = process_text(abstract) if abstract else {
            "entities": [], "matched_terms": {}, "chunks": [], "embeddings": np.zeros((0, 384))
        }

    paper_id = generate_unique_id()  # Geração do ID principal do artigo
    topic = infer_topic_from_text(paper_data["title"], abstract)
    source_levels = {
            "PubMed": 2,
            "Europe PMC": 3,
            "Wikipedia": 3,
            "Semantic Scholar": 4,
            "Google Scholar": 4,
            "EatRight": 5,
            "Dietary Guidelines": 5
        }
    hierarchical_level = source_levels.get(paper_data["source"], 2)
    link = paper.get("url", "") or paper.get("link", "") or paper.get("doi", "")

    doc = {
        "paper_id": paper_id,  # O ID é o mesmo para o MongoDB
        "title": paper_data["title"],
        "authors": paper_data["authors"],
        "year": paper_data["year"],
        "source": paper_data["source"],
        "abstract": abstract,
        "keywords": paper_data["keywords"],
        "doi": paper_data["doi"],
        "journal": paper_data["journal"],
        "last_updated": paper_data["last_updated"],
        "spacy_entities": spacy_results["entities"][:200],
        "spacy_matched_terms": {key: values[:50] for key, values in spacy_results["matched_terms"].items()},
        "chunks": spacy_results["chunks"],
        "topic": topic,
        "hierarchical_level": hierarchical_level,
        "link": link,
    }
    collection.insert_one(doc)

    embeddings = spacy_results["embeddings"]
    chunks = spacy_results["chunks"]

    if len(embeddings) != len(chunks):
        print(f"Warning: Mismatch between embeddings ({len(embeddings)}) and chunks ({len(chunks)}) for paper {paper_id}")
        return

    ids_filename = "ids.json"
    try:
        with open(ids_filename, "r", encoding="utf-8") as f:
            existing_ids = set(json.load(f))
    except FileNotFoundError:
        existing_ids = set()

    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        if embedding.shape != (384,):
            print(f"Invalid embedding shape for chunk {i} of paper {paper_id}: {embedding.shape}")
            continue

        chunk_id = f"{paper_id}_chunk_{i}"  # O chunk_id inclui o paper_id

        # Construir campos obrigatórios do novo formato
        minimal_metadata = {
            "chunk_id": chunk_id,
            "chunk_text": chunk[:2000],
            "title": paper_data["title"],
            "link": paper.get("url", "") or paper.get("link", "") or paper.get("doi", ""),
            "year": str(paper_data["year"]),
            "topic": topic,
            "hierarchical_level": hierarchical_level
        }

        # Guardar no Pinecone
        full_metadata = {
            **minimal_metadata,
            "paper_id": paper_id,  # Garantir que o paper_id no Pinecone é o mesmo do MongoDB
            "chunk_idx": i,
            "source": paper_data["source"],
            "doi": paper_data["doi"]
        }

        index.upsert(vectors=[(chunk_id, embedding.tolist(), full_metadata)])
        # Adicionar o ID à lista
        existing_ids.add(chunk_id)
    
    with open(ids_filename, "w", encoding="utf-8") as f:
        json.dump(list(existing_ids), f, indent=4, ensure_ascii=False)


def save_to_mongo_and_pinecone(papers, source):
    """Save articles to MongoDB and Pinecone."""
    if not papers:
        print("No articles to save.")
        return

    collection = configure_mongoDB_connection()
    index = configure_pinecone_connection()

    for paper in tqdm(papers, desc=f"Saving {source} articles"):
        save_paper_to_mongo_and_pinecone(paper, source, collection, index)

    print(f"All {source} articles have been successfully saved!")


def infer_topic_from_text(title, abstract):
    text = f"{title} {abstract}".lower()

    topic_keywords = {
        "diabetes": ["diabetes", "insulin", "blood sugar", "glucose"],
        "cardiovascular health": ["heart", "cardiovascular", "cholesterol", "blood pressure"],
        "obesity": ["obesity", "overweight", "bmi", "body mass"],
        "nutrition": ["nutrition", "diet", "micronutrient", "macronutrient"],
        "gut health": ["gut", "microbiome", "digestive"],
        "pregnancy": ["pregnancy", "maternal", "prenatal", "gestation"],
        "cancer": ["cancer", "tumor", "oncology"],
        "mental health": ["depression", "anxiety", "mental", "cognition", "cognitive"],
        "supplements": ["supplement", "vitamin", "mineral", "omega-3", "iron", "zinc"],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            return topic

    return "general"


