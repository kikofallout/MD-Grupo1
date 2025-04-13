from modules.mongoDB_utils import configure_mongoDB_connection, configure_pinecone_connection
import pandas as pd
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # Dimensão: 384

# Conexões
collection = configure_mongoDB_connection()
pinecone_index = configure_pinecone_connection()

# Carregar CSV
csv_path = "files/eatright_articles.csv" 
df = pd.read_csv(csv_path)

# Função para gerar dados para Pinecone
def generate_pinecone_data(row):
    chunk_id = str(uuid.uuid4())
    
    chunk_text = row['content']
    
    title = row['title']
    url = row['url']
    
    source = "EatRight"
    topic = row['section'].lower()  

    # Gerar o dado no formato Pinecone
    return {
        "id": f"{chunk_id}_chunk",  
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "doi": "sem_doi",  
        "hierarchical_level": 2,
        "link": url,
        "paper_id": str(uuid.uuid4()),  
        "source": source,
        "title": title,
        "topic": topic,
        "year": str(datetime.now().year)  
    }

# Função para inserir no MongoDB
def insert_into_mongo(row):
    document = {
        "title": row['title'],
        "url": row['url'],
        "metadata": row['metadata'],
        "content": row['content'],
        "section": row['section'],
        "source": "EatRight",
        "year": datetime.now().year,
        "doi": "sem_doi",  # Como o CSV não contém DOI, fica como "sem_doi"
        "spacy_matched_terms": {}  
    }
    
    # Inserir no MongoDB
    collection.insert_one(document)

for index, row in df.iterrows():
    insert_into_mongo(row)
    
    pinecone_data = generate_pinecone_data(row)
    
    #index.upsert([(pinecone_data["id"], [0.0] * 384, pinecone_data)])  
    #pinecone_index.upsert([(pinecone_data["id"], [0.0] * 384, pinecone_data)])
    embedding = model.encode(pinecone_data["chunk_text"]).tolist()
    pinecone_index.upsert([(pinecone_data["id"], embedding, pinecone_data)])



print("Dados inseridos com sucesso!")
