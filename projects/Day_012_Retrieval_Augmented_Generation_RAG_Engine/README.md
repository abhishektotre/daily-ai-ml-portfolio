# Day 12: Retrieval-Augmented Generation (RAG) Architecture with Vector Indexing

![Domain](https://img.shields.io/badge/Domain-Artificial%20Intelligence-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Retrieval-Augmented Generation (RAG) grounds language systems in factual, verified external knowledge bases to eliminate hallucinations. This project provides:
1. Multi-document knowledge base ingestion with sliding-window chunking.
2. Dense semantic vector representation using TF-IDF and normalized embeddings.
3. In-memory Vector Database supporting top-$k$ nearest neighbor query matching.
4. Relevance score calibration & reciprocal rank scoring.
5. Context injection engine that constructs grounded synthesis prompts with citation metadata.

## 🛠️ Project Structure
```text
Day_012_Retrieval_Augmented_Generation_RAG_Engine/
├── data/
│   └── knowledge_base.json
├── results/
│   ├── query_similarity_scores.png
│   └── rag_query_responses.json
├── src/
│   ├── __init__.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_012_Retrieval_Augmented_Generation_RAG_Engine
pip install -r requirements.txt
python main.py
```
