import spacy
import json
from spacy.matcher import PhraseMatcher
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Função para carregar termos de um arquivo JSON
def load_terms_from_json(file_path):
     with open(file_path, "r") as file:
         data = json.load(file)
     # Retorna a lista de termos a partir da chave correspondente
     return set(data[list(data.keys())[0]])

# Verificar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Carregar o modelo SciBERT do spaCy
#nlp = spacy.load("en_core_sci_scibert")
nlp = spacy.load("en_core_web_trf")

# Carregar o modelo de embeddings BGE da HuggingFace
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")
model = AutoModel.from_pretrained("BAAI/bge-small-en")
model.to(device)

# Função para normalizar texto
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Carregar os termos de cada JSON
food_terms = load_terms_from_json("terms/food_terms.json")
nutrient_terms = load_terms_from_json("terms/nutrient_terms.json")
diet_terms = load_terms_from_json("terms/diet_terms.json")
eating_habits_terms = load_terms_from_json("terms/eating_habits_terms.json")

# Normalizar todos os termos
food_terms = set(normalize_text(term) for term in food_terms)
nutrient_terms = set(normalize_text(term) for term in nutrient_terms)
diet_terms = set(normalize_text(term) for term in diet_terms)
eating_habits_terms = set(normalize_text(term) for term in eating_habits_terms)

# Função para criar um PhraseMatcher
def create_matcher(nlp, terms):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add("TERM_MATCHER", patterns)
    return matcher

# Criar matchers
food_matcher = create_matcher(nlp, food_terms)
nutrient_matcher = create_matcher(nlp, nutrient_terms)
diet_matcher = create_matcher(nlp, diet_terms)
eating_habits_matcher = create_matcher(nlp, eating_habits_terms)

# Função para dividir o texto em chunks
def split_into_chunks(text, max_length=150):  # Ajustado para RAG
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_tokens = len(tokenizer.tokenize(sent_text))
        if sent_tokens > max_length:
            words = sent_text.split()
            sub_chunk = []
            sub_length = 0
            for word in words:
                word_tokens = len(tokenizer.tokenize(word))
                if sub_length + word_tokens <= max_length:
                    sub_chunk.append(word)
                    sub_length += word_tokens
                else:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_length = word_tokens
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        else:
            if current_length + sent_tokens <= max_length:
                current_chunk.append(sent_text)
                current_length += sent_tokens
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_length = sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Função para gerar embeddings
def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.zeros((0, 384))  # Return [n_chunks, 384] array

# Função para processar o texto
def process_text(text):
    normalized_text = normalize_text(text)
    doc = nlp(normalized_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    matches = {
        "FOOD": [doc[start:end].text for _, start, end in food_matcher(doc)],
        "NUTRIENT": [doc[start:end].text for _, start, end in nutrient_matcher(doc)],
        "DIET": [doc[start:end].text for _, start, end in diet_matcher(doc)],
        "EATING_HABITS": [doc[start:end].text for _, start, end in eating_habits_matcher(doc)]
    }
    categorized_entities = []
    for ent_text, ent_label in entities:
        for category, matched_terms in matches.items():
            if ent_text in matched_terms:
                categorized_entities.append((ent_text, category))
                break
    chunks = split_into_chunks(text)
    embeddings = generate_embeddings(chunks)  # Should return [n_chunks, 384]
    return {
        "entities": categorized_entities,
        "matched_terms": matches,
        "chunks": chunks,
        "embeddings": embeddings
    }
