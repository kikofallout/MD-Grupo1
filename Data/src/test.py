import os
from dotenv import load_dotenv
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import pinecone
from modules.spaCy_utils import generate_embeddings, split_into_chunks

def configure_pinecone_connection():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key= api_key)
    index_name = "papers"
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' não encontrado.")
    return pc.Index(index_name)

def query_pinecone(query_text, index, top_k=5):
    """Query Pinecone for the top-k most similar chunks."""
    # Generate query embedding
    query_chunks = split_into_chunks(query_text)
    query_embedding = generate_embeddings(query_chunks)
    if query_embedding.shape[0] == 0:
        print("No query embedding generated.")
        return []
    # Use the first chunk's embedding for simplicity (or average if multiple)
    query_vector = query_embedding[0].tolist() if query_embedding.shape[0] > 0 else np.zeros(384).tolist()

    # Query Pinecone
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    retrieved_chunks = []
    for match in results['matches']:
        retrieved_chunks.append({
            "chunk_id": match['id'],
            "chunk_text": match['metadata']['chunk_text'],
            "score": match['score'],
            "title": match['metadata']['title'],
            "doi": match['metadata']['doi']
        })
    return retrieved_chunks

# Configurar o Pinecone
index = configure_pinecone_connection()

# Definir a consulta
query_text = "eating habits nutrition diet"

# Buscar os chunks mais relevantes
top_k = 5
retrieved_chunks = query_pinecone(query_text, index, top_k=top_k)

# Exibir os resultados
if not retrieved_chunks:
    print("Nenhum chunk encontrado para a consulta.")
else:
    print(f"Top {top_k} chunks mais relevantes para a consulta '{query_text}':")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n{i}. Chunk ID: {chunk['chunk_id']}")
        print(f"Score: {chunk['score']:.4f}")
        print(f"Title: {chunk['title']}")
        print(f"DOI: {chunk['doi']}")
        print(f"Chunk Text: {chunk['chunk_text']}")
        print("-" * 80)