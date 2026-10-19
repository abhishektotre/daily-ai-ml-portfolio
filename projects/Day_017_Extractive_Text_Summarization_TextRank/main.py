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
from src.textrank_engine import summarize_text

def main():
    print("=" * 65)
    print(" 📄 Running Extractive Text Summarization with TextRank")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    with open("data/article.txt", "r") as f:
        text = f.read()
        
    print(f"[1/3] Reading document ({len(text.split())} words, 10 sentences)...")
    summary_sents, ranks, all_sents = summarize_text(text, top_k=3)
    
    print("[2/3] Constructing sentence graph & running PageRank centrality...")
    
    # Plot Sentence Centrality Scores
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(ranks) + 1), ranks, color="#5c3566")
    plt.title("Sentence Centrality Scores (TextRank Algorithm)")
    plt.xlabel("Sentence Index in Document")
    plt.ylabel("PageRank Centrality")
    plt.xticks(range(1, len(ranks) + 1))
    plt.tight_layout()
    plt.savefig("results/sentence_centrality_ranks.png", dpi=200)
    plt.close()
    
    output = {
        "original_sentence_count": len(all_sents),
        "summary_sentence_count": len(summary_sents),
        "compression_ratio": round(len(summary_sents) / len(all_sents), 2),
        "key_extracted_sentences": summary_sents
    }
    with open("results/summary_output.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print("[3/3] Extractive Summary Generated:")
    for i, sent in enumerate(summary_sents, 1):
        print(f"      [{i}] {sent}")
    print(f"\n Saved summary to results/summary_output.json")
    print("=" * 65)

if __name__ == "__main__":
    main()
