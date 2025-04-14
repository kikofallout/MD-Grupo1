import os
import json
from tqdm import tqdm
from pinecone import Pinecone

def configure_pinecone_connection():
    """Configura a conexão com o Pinecone v3."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("data")  # Nome do índice
    return index

def export_all_pinecone_vectors_from_ids(ids_file="ids.json", output_file="pinecone_data_export.json"):
    index = configure_pinecone_connection()

    # Carrega os IDs
    try:
        with open(ids_file, "r", encoding="utf-8") as f:
            all_ids = json.load(f)
    except FileNotFoundError:
        print(f"Arquivo {ids_file} não encontrado.")
        return

    limit = 100
    namespace = ""
    all_vectors = []

    for i in tqdm(range(0, len(all_ids), limit), desc="Exportando vetores"):
        batch_ids = all_ids[i:i + limit]
        response = index.fetch(ids=batch_ids, namespace=namespace)

        for vector_id, vector_data in response["vectors"].items():
            metadata = vector_data.get("metadata", {})
            all_vectors.append({
                "id": vector_id,
                **metadata
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_vectors, f, indent=4, ensure_ascii=False)

    print(f"{len(all_vectors)} vetores exportados para {output_file}")

if __name__ == "__main__":
    export_all_pinecone_vectors_from_ids()
