import json
import os
from Bio import Entrez
from dotenv import load_dotenv
from modules.mongoDB_utils import save_to_mongo_and_pinecone
from modules.spaCy_utils import process_text

def fetch_papers(id_list):
    
    """Fetches article details from PubMed."""
    if not id_list:
        return []

    with Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="xml") as handle:
        records = Entrez.read(handle)

    results = []
    for article in records["PubmedArticle"]:
        medline = article["MedlineCitation"]
        article_data = medline["Article"]

        title = article_data.get("ArticleTitle", "No Title Available")
        article_date = article_data.get("ArticleDate", [])
        if article_date:
            year = article_date[0].get("Year", "No Year Available")
        else:
            year = "No Year Available"
        abstract_data = article_data.get("Abstract", {}).get("AbstractText", ["No Abstract"])
        abstract = " ".join(abstract_data) if isinstance(abstract_data, list) else str(abstract_data)
        keywords = [kw for sublist in medline.get("KeywordList", []) for kw in sublist] or ["No Keywords"]

        authors = [
            f"{author['ForeName']} {author['LastName']}"
            for author in article_data.get("AuthorList", [])
            if "LastName" in author and "ForeName" in author
        ]

        journal = article_data.get("Journal", {}).get("Title", "No Journal Info")
        doi = next((eloc.lower() for eloc in article_data.get("ELocationID", []) if eloc.attributes.get("EIdType") == "doi"), "No DOI")

        spacy_results = process_text(abstract)

        results.append({
            "title": title,
            "year": year,
            "abstract": abstract,
            "keywords": keywords,
            "authors": authors,
            "journal": journal,
            "doi": doi,
            "spacy_entities": spacy_results["entities"],
            "spacy_matched_terms": spacy_results["matched_terms"]
        })

    return results



def search_pubmed(query, num_results, year_range=None):
    """Searches for articles on PubMed and saves them to MongoDB."""
    load_dotenv()
    configure_entrez()
    
    if year_range:
        start_year, end_year = year_range
        query += f" AND ({start_year}[PDAT] : {end_year}[PDAT])"

    with Entrez.esearch(db="pubmed", term=query, retmax=num_results, sort="pub_date") as handle:
        record = Entrez.read(handle)

    articles = fetch_papers(record.get("IdList", []))
    save_to_mongo_and_pinecone(articles, "PubMed")
    return articles


def save_results_to_json(articles, filename="pubmed_results.json"):
    """Saves the articles to a JSON file."""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(articles, file, ensure_ascii=False, indent=4)
    print(f"Results saved in {filename}")

def configure_entrez():
    """Configures Entrez using environment variables."""
    email = os.getenv("EMAIL")
    api_key = os.getenv("API_KEY_PUBMED")

    if not email or not api_key:
        raise ValueError("Missing EMAIL or API_KEY_PUBMED in environment variables.")

    Entrez.email = email
    Entrez.api_key = api_key
