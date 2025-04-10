import os
import uuid
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm
from modules.spaCy_utils import process_text
from pinecone import Pinecone, ServerlessSpec

def configure_mongoDB_connection():
    """Configure MongoDB connection."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["nutrition"]
    collection = db["papers"]
    return collection

def configure_pinecone_connection():
    """Configure Pinecone connection."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "papers"
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

    else:
        raise ValueError(f"Unsupported source: {source}")

def save_paper_to_mongo_and_pinecone(paper, source, collection, index):
    """Save a single paper to MongoDB and Pinecone."""
    # Extract paper attributes
    paper_data = extract_paper_attributes(paper, source)
    abstract = paper_data["abstract"]
    
    # Process the abstract with spaCy
    #spacy_results = process_text(abstract) if abstract else {
     #   "entities": [], "matched_terms": {}, "chunks": [], "embeddings": np.zeros((0, 384))
    #}
    spacy_results = {
        "entities": paper.get("spacy_entities", []),
        "matched_terms": paper.get("spacy_matched_terms", {}),
        "chunks": paper.get("chunks", []),
        "embeddings": paper.get("embeddings", [])
    }
    #if not spacy_results["chunks"] or not spacy_results["embeddings"]:
    #if len(spacy_results["chunks"]) == 0 or spacy_results["embeddings"].size == 0:
    if not spacy_results["chunks"] or spacy_results["embeddings"].shape[0] == 0:
        spacy_results = process_text(abstract) if abstract else {
            "entities": [], "matched_terms": {}, "chunks": [], "embeddings": np.zeros((0, 384))
        }

    # Generate a unique paper ID
    paper_id = generate_unique_id()
    
    # Create MongoDB document
    doc = {
        "paper_id": paper_id,
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
    }
    collection.insert_one(doc)
    
    # Save embeddings and chunk text to Pinecone
    embeddings = spacy_results["embeddings"]  # Shape: [n_chunks, 384]
    chunks = spacy_results["chunks"]
    if len(embeddings) != len(chunks):
        print(f"Warning: Mismatch between embeddings ({len(embeddings)}) and chunks ({len(chunks)}) for paper {paper_id}")
        return
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        if embedding.shape != (384,):
            print(f"Invalid embedding shape for chunk {i} of paper {paper_id}: {embedding.shape}")
            continue
        chunk_id = f"{paper_id}_chunk_{i}"
        metadata = {
            "paper_id": paper_id,
            "chunk_idx": i,
            "chunk_text": chunk[:2000],   # Store the chunk text in Pinecone metadata
            "title": paper_data["title"],
            "source": paper_data["source"],
            "year": paper_data["year"],
            "doi": paper_data["doi"]
        }
        index.upsert(vectors=[(chunk_id, embedding.tolist(), metadata)])

def save_to_mongo_and_pinecone(papers, source):
    """Save articles to MongoDB and Pinecone."""
    if not papers:
        print("No articles to save.")
        return
    
    # Configure connections
    collection = configure_mongoDB_connection()
    index = configure_pinecone_connection()
    
    # Process each paper
    for paper in tqdm(papers, desc=f"Saving {source} articles"):
        save_paper_to_mongo_and_pinecone(paper, source, collection, index)
    
    print(f"All {source} articles have been successfully saved!")