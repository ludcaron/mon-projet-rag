# Projet RAG Python avec Ollama

## Prérequis

- Python 3.8+
- [Ollama](https://ollama.com/download) installé et configuré  
- Les modèles téléchargés via `ollama pull mistral` ou `ollama pull phi3`
- Installer les dépendances Python :
  ```
  pip install -r requirements.txt
  ```

## Lancer un test simple avec Ollama

```
python ollama_simple_chat.py
```

## Lancer le RAG

```
python rag.py
```

## Où mettre vos documents ?

Ajoutez vos fichiers texte dans le dossier `data/`.

---