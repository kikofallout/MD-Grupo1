import os
import json
from pathlib import Path
import wikipediaapi
from modules.mongoDB_utils import save_to_mongo_and_pinecone

def load_terms_from_json_folder(folder_path):
    all_terms = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, terms in data.items():
                    if isinstance(terms, list):
                        all_terms.update(terms)
    return sorted(all_terms)

def search_wikipedia(query, lang="en"):
    user_agent = "NutriBot-KnowledgeBase/1.0 (NutriBot-KnowledgeBase; email@example.com)"
    wiki_wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent=user_agent
    )
    page = wiki_wiki.page(query)
    if page.exists():
        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.fullurl
        }
    else:
        return None

def search_all_terms_and_print(folder_path, lang="en"):
    terms = load_terms_from_json_folder(folder_path)
    print(f"🔍 {len(terms)} termos encontrados. Iniciando buscas na Wikipedia...\n")

    articles = []
    for term in terms:
        result = search_wikipedia(term, lang=lang)
        if result:
            print(f"✅ {result['title']}\n{result['summary'][:300]}...\n🔗 {result['url']}\n")
            articles.append(result)
        else:
            print(f"❌ Nenhuma página encontrada para: {term}\n")

    print(f"🔍 {len(articles)} artigos encontrados na Wikipedia.\n")
    if articles:
        print(f"\n💾 Guardando {len(articles)} artigos no MongoDB e Pinecone...\n")
        save_to_mongo_and_pinecone(articles, "Wikipedia")
    else:
        print("⚠️ Nenhum artigo válido encontrado.")

