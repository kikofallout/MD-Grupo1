from rich.console import Console
from rich.prompt import Prompt
from modules.europePMC_utils import search_europe_pmc
from modules.googleScholar_utils import search_google_scholar
from modules.menu_utils import display_menu
from modules.pubmed_utils import search_pubmed
from modules.semanticscholar_utils import search_semanticscholar
from modules.wikipedia_utils import search_wikipedia, load_terms_from_json_folder, search_all_terms_and_print
from modules.eatright_utils import process_pdf_links
from modules.dietaryguidelines_utils import process_dietary_pdfs
from modules.mongoDB_utils import configure_mongoDB_connection, save_paper_to_mongo_and_pinecone, configure_pinecone_connection
import time
import pandas as pd
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pathlib import Path
from sentence_transformers import SentenceTransformer
from modules.nih_utils import process_nih_articles
from modules.oms_utils import process_who_articles


def search_and_print(source, func, query, max_articles=1, year_range=(2020, 2025)):
    """Performs the search and prints the results."""
    console = Console()
    console.print(f"\n[bold cyan]🔎 Searching on {source}...[/bold cyan]")

    time.sleep(2)
    
    results = func(query) if source == "Wikipedia" else func(query, max_articles, year_range)
    
    if source == "Wikipedia" and results:
        console.print(f"\n[bold yellow]📖 Wikipedia: {results['title']}[/bold yellow]")
        console.print(f"🔗 [blue]{results['url']}[/blue]")
        console.print(f"📝 {results['summary']}")
    
    return results

def main():
    """Runs searches based on user choice."""
    console = Console()
    query = '"eating habits" AND "nutrition" AND ("health outcomes" OR "dietary patterns")'
    #"eating habits" AND "nutrition" AND ("health outcomes" OR "dietary patterns")
    #"dietary intake" AND "eating habits" AND "nutrition"
    #"nutritional status" AND "eating habits" AND ("public health" OR "chronic disease")
    #"nutrition education" AND "eating habits" AND ("prevention" OR "public health")
    #"healthy eating" AND "nutrition" AND ("policy" OR "intervention")
    #"plant-based diet" AND "eating behavior" AND ("health benefits" OR "disease prevention")
    #"dietary patterns" AND "nutritional intake" AND "chronic diseases"
    #"adolescents" AND "eating habits" AND "nutrition"
    #"elderly" AND "nutritional status" AND "dietary patterns"
    #"children" AND "nutrition" AND "eating behavior"




    sources = {
        "1": ("PubMed", search_pubmed),
        "2": ("Europe PMC", search_europe_pmc),
        "3": ("Semantic Scholar", search_semanticscholar),
        "4": ("Wikipedia", search_wikipedia),
        "5": ("Google Scholar", search_google_scholar),
        "6": ("EatRight", None),  
        "7": ("Dietary Guidelines", None),
        "8": ("NIH", process_nih_articles),
        "9": ("OMS", process_who_articles),
        "10": ("All Sources", None),
        "11": ("EatRight csv", None)
    }
    
    while True:
        choice = display_menu()
        if choice.lower() == 'q':
            console.print("\n[bold red]🚪 Exiting...[/bold red]")
            break
        elif choice in sources:
            if choice == "6":
                console.print(f"\n[bold cyan]🔎 Extracting articles from EatRight...[/bold cyan]")
                #links de pdfs disponíveis no EatRight, tem de ser assim "caseiro" por causa do html do site
                pdf_urls = [
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/eating-right-and-reduce-food-waste.pdf?rev=a67aa98dd0f34c33a85b71f653d701bc&hash=4573EEAD91EBC5655768420E43B3A446",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/eating-right-on-a-budget.pdf?rev=c32b1a4280754a5eafa05b6171b14eb2&hash=241DA355844833CB7B3A07B144BD9619",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/eating-right-tips-for-older-adults.pdf?rev=195456ecc4d446959db35c7c199d621e&hash=2972E43BD04D99242858D6A2D3CA7ABA",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/food-connects-us.pdf?rev=acf5d10dcc3e457ab2e478562872d7c4&hash=AEAAD1615FDA2382846E238ABBAF7849",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/healthy-eating-on-the-run.pdf?rev=f1cf016c2e314282a918b40f3877a358&hash=F4AD9194F0B694A61F650968CAE04F42",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/smart-snacking-tips-for-adults-and-teens.pdf?rev=3587d9e1cbb8440e88d46a6c5ede87b4&hash=3E7680DD19D868CA9409187D3F7FB29A",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/smart-snacking-tips-for-kids.pdf?rev=25cd31e833f34ee38ca86615cc8ff53c&hash=932210736B84428018E473F895ABFC77",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/plant-based-eating-tip-sheet.pdf?rev=7119cc302de543e6a66fdd9bb4ba756c&hash=3A239ED0DDE5D712214D65164240D9EF",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/tip-sheets/smart-tips-for-successful-meals.pdf?rev=e1aa5ff5f76c4ad6b9126c5223bb009a&hash=C4FFC7FEB29A0CC323E1908039CAC592",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/activity-handouts/fact-or-fiction-handout.pdf?rev=eac51374f4834254a142d3ea90aac20b&hash=3663A0B69DD24D13A87CDB4AEC3084EA",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/activity-handouts/word-search.pdf?rev=1d5dce20176a46fab6492acc2e3e83f2&hash=49185C941D55DC829B69E503493EF261",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/proclamations/national-nutrition-month-2025-proclamation.docx?rev=320161ed92ff4d04bf9357d8fc5fc244&hash=04E9B7DEA079C0C39B01E90BC8870053",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/proclamations/nutrition-and-dietetics-technician-registered-day-2025-proclamation.docx?rev=adb58f02f8cb4682bfe39eac06b9261e&hash=763CE81D49616752CBF6B08C2E883348",
                    "https://www.eatright.org/-/media/files/campaigns/eatright/nnm/english/proclamations/registered-dietitian-nutritionist-day-2025-proclamation.docx?rev=635b034f405a4711818679307a7dc30e&hash=D8013A0FDB3A84D98E01DBB79FB90D54",
                ]
                process_pdf_links(pdf_urls) 
                console.print("[bold green]✔️ Articles from EatRight have been saved to MongoDB and Pinecone![/bold green]")
            
            elif choice == "7":
                console.print(f"\n[bold cyan]🔎 Extracting articles from Dietary Guidelines...[/bold cyan]")
                process_dietary_pdfs()
                console.print("[bold green]✔️ Articles from Dietary Guidelines have been saved to MongoDB and Pinecone![/bold green]")
            
            elif choice == "8":
                console.print(f"\n[bold cyan]🔎 Extracting articles from NIH...[/bold cyan]")
                nih_articles = process_nih_articles()
                if nih_articles:
                    console.print(f"[bold green]✔️ {len(nih_articles)} NIH articles saved to MongoDB and Pinecone![/bold green]")
                else:
                    console.print("[bold yellow]⚠️ No NIH articles found.[/bold yellow]")

            elif choice == "9":
                console.print(f"\n[bold cyan]🔎 Extracting articles from WHO...[/bold cyan]")
                process_who_articles()
                console.print("[bold green]✔️ WHO articles saved.[/bold green]")

            elif choice == "10":
                max_articles = int(Prompt.ask("[bold white]How many articles per source?[/bold white]", default="1"))
                for key, (source_name, search_func) in sources.items():
                    if key != "4" and key != "7":  # Exclude Wikipedia and All Sources
                        search_and_print(source_name, search_func, query, max_articles)
            elif choice == "11":
                console.print(f"\n[bold cyan]📤 Exporting and formatting EatRight CSV data...[/bold cyan]")
                
                # Conexões
                collection = configure_mongoDB_connection()
                pinecone_index = configure_pinecone_connection()
                model = SentenceTransformer("BAAI/bge-small-en-v1.5")

                # Carregar CSV
                csv_path = "files/eatright_articles.csv"
                df = pd.read_csv(csv_path)

                for index, row in df.iterrows():
                    # Montar o dicionário no formato padrão de entrada
                    paper = {
                        "title": row['title'],
                        "url": row['url'],
                        "metadata": row.get('metadata', {}),
                        "content": row['content'],
                        "section": row['section'],
                        "abstract": row['content'],  # Usamos o content como abstract
                        "keywords": [],
                        "authors": "",
                        "year": datetime.now().year,
                        "spacy_matched_terms": {},  # Pode ser preenchido no processamento
                    }

                    # Opcional: você pode passar `paper["spacy_entities"]`, etc. também se tiver
                    save_paper_to_mongo_and_pinecone(paper, "EatRight", collection, pinecone_index)

                console.print("[bold green]✔️ EatRight CSV data successfully exported to MongoDB and Pinecone![/bold green]")

                
            else:
                source_name, search_func = sources[choice]
                if source_name == "Wikipedia":
                    #query = Prompt.ask("[bold white]Enter a search term:[/bold white]")
                    #search_and_print(source_name, search_func, query)
                    json_folder_path = Path("terms")  
                    search_all_terms_and_print(json_folder_path)
                        
                else:
                    max_articles = int(Prompt.ask("[bold white]How many articles?[/bold white]", default="1"))
                    search_and_print(source_name, search_func, query, max_articles)
        else:
            console.print("\n[bold red]❌ Invalid choice! Please select a valid option.[/bold red]")

if __name__ == "__main__":
    main()
