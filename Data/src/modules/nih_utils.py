import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import re
from modules.mongoDB_utils import save_to_mongo_and_pinecone
import json
from datetime import datetime

def search_nih_nutrition_articles():
    """
    Searches and extracts nutrition-related articles from NIH websites.
    """
    articles = []
    
    # NIH nutrition-related URLs from different institutes
    nih_urls = [
        # NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)
        "https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity",
        "https://www.niddk.nih.gov/health-information/weight-management/healthy-eating-physical-activity-for-life",
        "https://www.niddk.nih.gov/health-information/digestive-diseases/constipation/eating-diet-nutrition",
        
        # NHLBI (National Heart, Lung, and Blood Institute)
        "https://www.nhlbi.nih.gov/health/educational/lose_wt/eat/calories.htm",
        
        # NCI (National Cancer Institute)
        "https://www.cancer.gov/about-cancer/causes-prevention/risk/diet",
        "https://www.cancer.gov/publications/patient-education/eating-hints",
        
        # ODS (Office of Dietary Supplements)
        "https://ods.od.nih.gov/factsheets/list-all/",
        
        # NIH News
        "https://www.nih.gov/news-events/news-releases?topic_id%5B%5D=46",  # Nutrition topic
    ]
    
    for url in nih_urls:
        try:
            if "factsheets/list-all" in url:
                # Special handling for ODS fact sheets
                factsheet_articles = extract_ods_factsheets()
                articles.extend(factsheet_articles)
            elif "news-releases" in url:
                # Special handling for news releases
                news_articles = extract_nih_news(url)
                articles.extend(news_articles)
            else:
                article_data = extract_nih_article(url)
                if article_data:
                    articles.append(article_data)
            time.sleep(2)  # Be respectful to NIH servers
        except Exception as e:
            print(f"Error processing NIH URL {url}: {str(e)}")
            continue
    
    return articles

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_nih_article(url):
    """
    Extracts content from a NIH article page.
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
            'h1.page-title',
            'h1#page-title',
            'h1',
            '.entry-title',
            '.article-title',
            '.content-title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                break
        
        # Extract main content
        content = ""
        content_selectors = [
            '.field--name-body',
            '.node__content',
            '.main-content',
            '#main-content',
            '.content-body',
            'main',
            '.article-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove unwanted elements
                for unwanted in content_elem(["script", "style", "nav", "header", "footer", ".sidebar", ".navigation"]):
                    unwanted.decompose()
                content = content_elem.get_text(separator=' ', strip=True)
                break
        
        # Clean content
        content = re.sub(r'\s+', ' ', content)
        content = content.replace('\n', ' ').replace('\t', ' ')
        
        # Extract publication date
        pub_date = extract_nih_date(soup)
        
        # Extract institute information
        institute = extract_nih_institute(url, soup)
        
        if title and content and len(content) > 200:
            full_text = f"{title}. {content}"
            chunks = split_text_into_chunks(full_text)
            return {
                'title': title,
                'abstract': content[:2000],  # Limit abstract length
                'content': content,
                'url': url,
                'authors': f"National Institutes of Health - {institute}",
                'year': pub_date.year if pub_date else 2024,
                'last_updated': pub_date.strftime('%Y-%m-%d') if pub_date else "",
                'keywords': extract_nih_keywords(title, content),
                'institute': institute,
                'source': 'NIH',
                'chunks': chunks 
            }
    
    except Exception as e:
        print(f"Error extracting NIH article from {url}: {str(e)}")
        return None

def extract_ods_factsheets():
    """
    Extracts fact sheets from the Office of Dietary Supplements.
    """
    articles = []
    base_url = "https://ods.od.nih.gov"
    
    # Key dietary supplement fact sheets
    factsheet_urls = [
        "/factsheets/VitaminD-Consumer/",
        "/factsheets/VitaminC-Consumer/",
        "/factsheets/Calcium-Consumer/",
        "/factsheets/Iron-Consumer/",
        "/factsheets/VitaminB12-Consumer/",
        "/factsheets/Folate-Consumer/",
        "/factsheets/Omega3FattyAcids-Consumer/",
        "/factsheets/Magnesium-Consumer/",
        "/factsheets/Zinc-Consumer/",
        "/factsheets/VitaminA-Consumer/"
    ]
    
    for factsheet_url in factsheet_urls:
        try:
            full_url = base_url + factsheet_url
            article_data = extract_nih_article(full_url)
            if article_data:
                articles.append(article_data)
            time.sleep(1)
        except Exception as e:
            print(f"Error processing ODS factsheet {factsheet_url}: {str(e)}")
            continue
    
    return articles

def extract_nih_news(news_url, max_articles=5):
    """
    Extracts recent nutrition-related news from NIH.
    """
    articles = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(news_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find news article links
        article_links = []
        link_selectors = [
            '.views-row a',
            '.news-item a',
            '.list-item a',
            'h3 a',
            'h2 a'
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links[:max_articles]:
                href = link.get('href')
                title_text = link.get_text(strip=True)
                if href and is_nutrition_related(title_text):
                    full_url = urljoin('https://www.nih.gov', href)
                    article_links.append(full_url)
        
        # Extract content from each news article
        for link in article_links[:max_articles]:
            try:
                article_data = extract_nih_article(link)
                if article_data:
                    articles.append(article_data)
                time.sleep(1)
            except Exception as e:
                print(f"Error processing NIH news article {link}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error processing NIH news: {str(e)}")
    
    return articles

def extract_nih_date(soup):
    """
    Extracts publication date from NIH page.
    """
    date_selectors = [
        '.field--name-field-date-posted',
        '.date-posted',
        '.publication-date',
        '.last-updated',
        'meta[name="article:published_time"]',
        'meta[property="article:published_time"]',
        'time[datetime]'
    ]
    
    for selector in date_selectors:
        date_elem = soup.select_one(selector)
        if date_elem:
            date_text = date_elem.get('content') or date_elem.get('datetime') or date_elem.get_text(strip=True)
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%B %d, %Y', '%m/%d/%Y', '%Y']:
                    try:
                        return datetime.strptime(date_text.strip(), fmt)
                    except ValueError:
                        continue
            except:
                pass
    
    return None

def extract_nih_institute(url, soup):
    """
    Determines which NIH institute the content comes from.
    """
    url_lower = url.lower()
    
    institute_mapping = {
        'niddk': 'NIDDK',
        'nhlbi': 'NHLBI',
        'cancer.gov': 'NCI',
        'nichd': 'NICHD',
        'ods.od': 'ODS',
        'nccih': 'NCCIH',
        'nia': 'NIA'
    }
    
    for key, institute in institute_mapping.items():
        if key in url_lower:
            return institute
    
    # Try to extract from page content
    institute_selectors = [
        '.site-name',
        '.logo-text',
        'meta[name="DC.publisher"]'
    ]
    
    for selector in institute_selectors:
        elem = soup.select_one(selector)
        if elem:
            text = elem.get('content') or elem.get_text(strip=True)
            for key, institute in institute_mapping.items():
                if key.upper() in text.upper():
                    return institute
    
    return 'NIH'

def extract_nih_keywords(title, content):
    """
    Extracts keywords from NIH content.
    """
    nutrition_keywords = [
        'nutrition', 'diet', 'food', 'health', 'vitamin', 'mineral',
        'supplement', 'dietary', 'eating', 'nutrient', 'protein',
        'carbohydrate', 'fat', 'fiber', 'calcium', 'iron', 'folate',
        'omega-3', 'antioxidant', 'diabetes', 'obesity', 'heart disease',
        'cancer prevention', 'bone health', 'immune system'
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
        'nutrition', 'diet', 'food', 'eating', 'vitamin', 'mineral',
        'supplement', 'dietary', 'nutrient', 'health', 'obesity',
        'diabetes', 'heart', 'cancer', 'bone', 'immune'
    ]
    
    text_lower = text.lower()
    return any(term in text_lower for term in nutrition_terms)

def process_nih_articles():
    """
    Main function to process NIH articles and save to MongoDB/Pinecone.
    """
    print("Extracting articles from NIH...")
    articles = search_nih_nutrition_articles()
    
    if articles:
        print(f"Found {len(articles)} NIH articles")
        save_to_mongo_and_pinecone(articles, "NIH")
        print("NIH articles successfully saved to MongoDB and Pinecone!")
    else:
        print("No NIH articles found")
    
    return articles