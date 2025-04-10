import wikipediaapi

def search_wikipedia(query, lang="en"):
    """
    Search for a topic on Wikipedia and return a summary.
    
    Args:
        query (str): The topic to search for.
        lang (str): The language of the Wikipedia (default is "en" for English).
    
    Returns:
        dict: A dictionary containing the title, summary, and URL of the page.
              Returns None if the page does not exist.
    """
    # Defina um User-Agent único para o seu aplicativo
    user_agent = "NutriBot-KnowledgeBase/1.0 (NutriBot-KnowledgeBase; email@example.com)"
    
    # Crie o objeto Wikipedia com o User-Agent e o idioma
    wiki_wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent=user_agent
    )
    
    # Busque a página
    page = wiki_wiki.page(query)
    
    if page.exists():
        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.fullurl
        }
    else:
        return None