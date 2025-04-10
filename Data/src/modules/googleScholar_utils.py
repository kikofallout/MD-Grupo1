import json
import os
from dotenv import load_dotenv
from modules.mongoDB_utils import save_to_mongo_and_pinecone
from modules.spaCy_utils import process_text
from scholarly import scholarly

def fetch_papers(query, num_results, year_range=None):
    """Fetches article details from Google Scholar using scholarly."""
    if not query:
        return []

    search_query = scholarly.search_pubs(query)  # This returns a generator, not a list.
    
    results = []
    count = 0

    while count < num_results:
        try:
            paper = next(search_query)
            # Extracting paper details
            bib_data = paper.get("bib", {})
            title = bib_data.get("title", "No Title Available")
            authors = bib_data.get("author", "No Authors Available")
            year = bib_data.get("pub_year", "No Year Available")
            journal = bib_data.get("journal", "No Journal Info")
            abstract = bib_data.get("abstract", "No Abstract Available")
            doi = bib_data.get("doi", "No DOI")
            
            # Convert year to integer if possible, and apply year range filter if provided
            try:
                year = int(year)
            except ValueError:
                year = 0
            
            if year_range and (year < year_range[0] or year > year_range[1]):
                continue  # Skip if outside the range
            
            # Process the abstract with spaCy
            spacy_results = process_text(abstract)

            results.append({
                "title": title,
                "year": year,
                "abstract": abstract,
                "authors": authors if isinstance(authors, list) else [authors],  # Ensure authors are a list
                "journal": journal,
                "doi": doi,
                "spacy_entities": spacy_results["entities"],
                "spacy_matched_terms": spacy_results["matched_terms"]
            })

            count += 1

        except StopIteration:
            break

    return results


def search_google_scholar(query, num_results, year_range=None):
    """Searches for articles on Google Scholar and saves them to MongoDB."""
    # Search the papers using scholarly
    papers = fetch_papers(query, num_results, year_range)
    
    if not papers:
        print("No articles found for the query.")
    else:
        # Save to MongoDB
        save_to_mongo_and_pinecone(papers, "GoogleScholar")
        print(f"{len(papers)} articles saved to MongoDB.")

    return papers
