import time
import requests
from modules.mongoDB_utils import save_to_mongo_and_pinecone

def fetch_papers(query, num_results, max_retries=5, delay_between_retries=10):
    """Fetch articles from the Semantic Scholar API with retry and rate-limiting handling."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": num_results,
        "fields": "title,authors,year,abstract,journal,externalIds"
    }
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json().get("data", [])
            
            elif response.status_code == 429:  # Rate limit exceeded
                print(f"Rate limit exceeded. Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)  # Wait before retrying
                retries += 1
            else:
                print(f"Error fetching data: {response.status_code}")
                return []
        
        except requests.exceptions.RequestException as e:
            print(f"Error during request: {e}")
            return []
    
    print("Max retries reached. Could not fetch data.")
    return []

def search_semanticscholar(query, num_results, year_range=None):
    """Fetch articles from the Semantic Scholar API and save them to MongoDB."""
    papers = fetch_papers(query, num_results)
    
    if papers:
        save_to_mongo_and_pinecone(papers, "Semantic Scholar")
        print(f"{len(papers)} papers saved to MongoDB.")
    else:
        print("No papers found or failed to fetch papers.")
    
    return papers
