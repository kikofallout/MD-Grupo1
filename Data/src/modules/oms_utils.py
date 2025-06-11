import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import re
from modules.mongoDB_utils import save_to_mongo_and_pinecone
import fitz  # PyMuPDF for PDF processing
from io import BytesIO

def search_who_nutrition_articles():
    """
    Searches and extracts nutrition-related articles from WHO website.
    """
    articles = []
    
    # WHO nutrition and diet related URLs
    who_urls = [
        "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "https://www.who.int/news-room/fact-sheets/detail/malnutrition",
        "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
        "https://www.who.int/health-topics/nutrition",
        "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases",
    ]
    
    # Also search for recent nutrition-related news
    search_urls = [
        "https://www.who.int/news?healthtopics=c3615d5b-2c2f-43ad-9076-c42a96a3e4cd",  # Nutrition topic
    ]
    
    for url in who_urls:
        try:
            article_data = extract_who_article(url)
            if article_data:
                articles.append(article_data)
            time.sleep(2)  # Be respectful to WHO servers
        except Exception as e:
            print(f"Error processing WHO URL {url}: {str(e)}")
            continue
    
    # Extract additional articles from search results
    for search_url in search_urls:
        try:
            search_articles = extract_who_search_results(search_url)
            articles.extend(search_articles)
            time.sleep(2)
        except Exception as e:
            print(f"Error processing WHO search URL {search_url}: {str(e)}")
            continue
    
    return articles

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def extract_who_article(url):
    """
    Extracts content from a WHO article page.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = ""
        title_selectors = [
            'h1.dynamic-content__heading',
            'h1.page-heading',
            'h1',
            '.sf-heading'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        # Extract main content
        content = ""
        content_selectors = [
            '.sf-content-block__content',
            '.dynamic-content__content',
            '.page-content',
            'main',
            '.content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove script and style elements
                for script in content_elem(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                content = content_elem.get_text(separator=' ', strip=True)
                break
        
        # Clean content
        content = re.sub(r'\s+', ' ', content)
        content = content.replace('\n', ' ').replace('\t', ' ')
        
        # Extract publication date
        pub_date = extract_who_date(soup)
        
        # Extract key facts if available
        key_facts = extract_who_key_facts(soup)
        if key_facts:
            content = f"Key Facts: {key_facts}\n\n{content}"
        
        if title and content and len(content) > 200:
            full_text = f"{title}. {content}"
            chunks = split_text_into_chunks(full_text)

            return {
                'title': title,
                'abstract': content[:2000],  # Limit abstract length
                'content': content,
                'url': url,
                'authors': "World Health Organization",
                'year': pub_date.year if pub_date else 2024,
                'last_updated': pub_date.strftime('%Y-%m-%d') if pub_date else "",
                'keywords': extract_who_keywords(title, content),
                'source': 'OMS',
                'chunks': chunks 
            }
    
    except Exception as e:
        print(f"Error extracting WHO article from {url}: {str(e)}")
        return None

def extract_who_search_results(search_url, max_results=5):
    """
    Extracts articles from WHO search results pages.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    articles = []
    
    try:
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        article_links = []
        link_selectors = [
            'a[href*="/news/"]',
            'a[href*="/publications/"]',
            'a[href*="/fact-sheets/"]',
            '.list-view--content a',
            '.vertical-list-item a'
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links[:max_results]:
                href = link.get('href')
                if href:
                    full_url = urljoin('https://www.who.int', href)
                    if is_nutrition_related(link.get_text()):
                        article_links.append(full_url)
        
        # Extract content from each article link
        for link in article_links[:max_results]:
            try:
                article_data = extract_who_article(link)
                if article_data:
                    articles.append(article_data)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error processing WHO article {link}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error processing WHO search results: {str(e)}")
    
    return articles

def extract_who_date(soup):
    """
    Extracts publication date from WHO page.
    """
    from datetime import datetime
    
    date_selectors = [
        '.page-meta__item--date',
        '.publication-date',
        '.last-update',
        'meta[name="article:published_time"]',
        'time'
    ]
    
    for selector in date_selectors:
        date_elem = soup.select_one(selector)
        if date_elem:
            date_text = date_elem.get('content') or date_elem.get_text(strip=True)
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%d %B %Y', '%B %d, %Y', '%Y']:
                    try:
                        return datetime.strptime(date_text.strip(), fmt)
                    except ValueError:
                        continue
            except:
                pass
    
    return None

def extract_who_key_facts(soup):
    """
    Extracts key facts from WHO articles.
    """
    key_facts = []
    
    # Look for key facts sections
    key_facts_selectors = [
        '.key-facts ul li',
        '.highlight-facts li',
        'ul li:contains("Key facts")',
        '.fact-list li'
    ]
    
    for selector in key_facts_selectors:
        facts = soup.select(selector)
        for fact in facts:
            fact_text = fact.get_text(strip=True)
            if len(fact_text) > 10:
                key_facts.append(fact_text)
    
    return '; '.join(key_facts) if key_facts else ""

def extract_who_keywords(title, content):
    """
    Extracts keywords from WHO content.
    """
    nutrition_keywords = [
        'nutrition', 'diet', 'food', 'health', 'obesity', 'malnutrition',
        'vitamin', 'mineral', 'protein', 'carbohydrate', 'fat', 'fiber',
        'diabetes', 'cardiovascular', 'cancer', 'prevention', 'dietary',
        'eating', 'meal', 'nutrient', 'supplement', 'guidelines'
    ]
    
    text = f"{title} {content}".lower()
    found_keywords = []
    
    for keyword in nutrition_keywords:
        if keyword in text:
            found_keywords.append(keyword)
    
    return found_keywords

def is_nutrition_related(text):
    """
    Checks if text is nutrition-related.
    """
    nutrition_terms = [
        'nutrition', 'diet', 'food', 'eating', 'obesity', 'malnutrition',
        'vitamin', 'mineral', 'healthy eating', 'dietary', 'nutrient'
    ]
    
    text_lower = text.lower()
    return any(term in text_lower for term in nutrition_terms)

def process_who_articles():
    """
    Main function to process WHO articles and save to MongoDB/Pinecone.
    """
    print("Extracting articles from WHO...")
    articles = search_who_nutrition_articles()
    
    if articles:
        print(f"Found {len(articles)} WHO articles")
        save_to_mongo_and_pinecone(articles, "OMS")
        print("WHO articles successfully saved to MongoDB and Pinecone!")
    else:
        print("No WHO articles found")
    
    return articles