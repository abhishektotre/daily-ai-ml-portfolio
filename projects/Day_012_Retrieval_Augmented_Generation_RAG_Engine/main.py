import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.vector_store import InMemoryVectorStore
from src.rag_pipeline import RAGPipeline

def main():
    print("=" * 65)
    print(" 🤖 Running Retrieval-Augmented Generation (RAG) Architecture")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Load Knowledge Base
    with open("data/knowledge_base.json", "r") as f:
        docs = json.load(f)
    print(f"[1/3] Indexing {len(docs)} documents into in-memory vector store...")
    
    vstore = InMemoryVectorStore()
    vstore.index_documents(docs)
    
    # 2. Execute RAG Queries
    rag = RAGPipeline(vstore)
    test_queries = [
        "How do residual connections prevent vanishing gradients in deep networks?",
        "What is the mathematical formulation of attention in Transformers?",
        "How does RLHF align model outputs with human reward models?"
    ]
    
    print("[2/3] Executing semantic search and context synthesis...")
    results = []
    query_labels = []
    top_scores = []
    
    for q in test_queries:
        resp = rag.query(q, top_k=2)
        results.append(resp)
        score = resp["citations"][0]["relevance_score"] if resp["citations"] else 0.0
        query_labels.append(q[:25] + "...")
        top_scores.append(score)
        print(f"\n❓ Query: {q}")
        print(f"📖 Top Hit: {resp['citations'][0]['title']} (Score: {score})")
        print(f"💡 Synthesized Answer: {resp['synthesized_response'][:130]}...")
        
    # Plot Similarity Scores
    plt.figure(figsize=(7, 4))
    plt.barh(query_labels, top_scores, color="#e69f00")
    plt.xlabel("Cosine Similarity Score")
    plt.title("RAG Retrieval Relevance Scores by Query")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig("results/query_similarity_scores.png", dpi=200)
    plt.close()
    
    with open("results/rag_query_responses.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n[3/3] RAG Execution Complete! Saved responses to results/rag_query_responses.json")
    print("=" * 65)

if __name__ == "__main__":
    main()
