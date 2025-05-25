import faiss
import numpy as np

def search_documents(query_embedding, embeddings, doc_texts, k=3):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    D, I = index.search(np.array([query_embedding]), k)
    return [doc_texts[i] for i in I[0]]