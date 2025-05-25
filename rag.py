from src.loader import load_documents
from src.embedder import get_embeddings, embed_query
from src.searcher import search_documents
import requests

# 1. Charger les documents
documents = load_documents("data")

# 2. Obtenir les embeddings des documents
embeddings, doc_texts = get_embeddings(documents)

# 3. Poser une question
question = input("Posez votre question : ")
query_embedding = embed_query(question)

# 4. Chercher les documents les plus pertinents
top_k_docs = search_documents(query_embedding, embeddings, doc_texts, k=3)

# 5. Construire le prompt pour Ollama
context = "\n".join(top_k_docs)
prompt = f"Voici des informations trouvées :\n{context}\n\nQuestion : {question}\nRéponds de façon concise en utilisant ces informations."

# 6. Appel au modèle Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": prompt
    }
)

full_response = ""
for line in response.iter_lines():
    if line:
        data = line.decode('utf-8')
        import json
        chunk = json.loads(data)
        full_response += chunk.get("response", "")

print("Réponse augmentée :")
print(full_response)
# print("\nRéponse augmentée :")
# print(response.json()["response"])