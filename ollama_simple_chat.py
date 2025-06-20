import requests

question = "Explique-moi ce qu'est le machine learning en une phrase simple."

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": question
    },
    stream=True  # <-- important pour lire ligne à ligne
)

# Pour récupérer toute la réponse complète :
full_response = ""
for line in response.iter_lines():
    if line:
        data = line.decode('utf-8')
        import json
        chunk = json.loads(data)
        full_response += chunk.get("response", "")

print("Réponse Ollama :")
print(full_response)