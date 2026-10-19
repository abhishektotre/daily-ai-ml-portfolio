import re
import numpy as np

def split_sentences(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
    return sentences

def clean_tokens(sentence):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
    stopwords = {"and", "the", "for", "with", "that", "this", "from", "are", "have", "been", "has", "into"}
    return [t for t in tokens if t not in stopwords]

def build_similarity_matrix(sentences):
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    token_sets = [set(clean_tokens(s)) for s in sentences]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                inter = len(token_sets[i].intersection(token_sets[j]))
                denom = np.log(len(token_sets[i]) + 1) + np.log(len(token_sets[j]) + 1)
                sim_matrix[i, j] = inter / (denom + 1e-9)
    return sim_matrix

def pagerank(sim_matrix, d=0.85, max_iter=100, tol=1e-6):
    n = sim_matrix.shape[0]
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = sim_matrix / row_sums
    
    ranks = np.ones(n) / n
    for _ in range(max_iter):
        new_ranks = (1 - d) / n + d * np.dot(P.T, ranks)
        if np.linalg.norm(new_ranks - ranks, 1) < tol:
            break
        ranks = new_ranks
    return ranks

def summarize_text(text, top_k=3):
    sentences = split_sentences(text)
    if len(sentences) <= top_k:
        return sentences, np.ones(len(sentences))
        
    sim_matrix = build_similarity_matrix(sentences)
    ranks = pagerank(sim_matrix)
    
    top_indices = ranks.argsort()[-top_k:][::-1]
    top_indices_sorted = sorted(top_indices)
    
    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return summary_sentences, ranks, sentences
