import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InMemoryVectorStore:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.vectors = None
        
    def index_documents(self, docs):
        self.documents = docs
        texts = [d["content"] for d in docs]
        self.vectors = self.vectorizer.fit_transform(texts)
        
    def search(self, query, top_k=2):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.vectors)[0]
        
        top_indices = sims.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score > 0:
                results.append({
                    "score": round(score, 4),
                    "document": self.documents[idx]
                })
        return results
