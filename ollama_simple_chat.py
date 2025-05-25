import requests

question = "Explique-moi ce qu'est le machine learning en une phrase simple."

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": question
    }
)
print("Réponse Ollama :")
print(response.json()["response"])