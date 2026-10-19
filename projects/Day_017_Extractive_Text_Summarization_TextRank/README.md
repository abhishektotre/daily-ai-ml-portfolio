# Day 17: Extractive Document Summarization with TextRank Algorithm

![Domain](https://img.shields.io/badge/Domain-NLP-purple)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Extractive summarization distills long documents into concise overviews by ranking sentences by information centrality. This project implements:
1. Sentence boundary segmentation and token-level vocabulary normalization.
2. Dense pairwise sentence similarity matrix construction based on token overlap & cosine distance.
3. Graph-based **PageRank (TextRank)** power iteration algorithm to identify the most salient nodes.
4. Top-$K$ key sentence extraction preserving chronological document flow.
5. Summary compression ratio and centrality distribution metrics.

## 🛠️ Project Structure
```text
Day_017_Extractive_Text_Summarization_TextRank/
├── data/
│   └── article.txt
├── results/
│   ├── sentence_centrality_ranks.png
│   └── summary_output.json
├── src/
│   ├── __init__.py
│   └── textrank_engine.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_017_Extractive_Text_Summarization_TextRank
pip install -r requirements.txt
python main.py
```
