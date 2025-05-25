from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_embeddings(documents):
    model = get_model()
    embeddings = model.encode(documents)
    return embeddings, documents

def embed_query(query):
    model = get_model()
    return model.encode([query])[0]