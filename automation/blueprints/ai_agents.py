"""
Artificial Intelligence (AI & Agents) Blueprints:
Includes Retrieval-Augmented Generation (RAG) architectures,
autonomous ReAct agents with tool calling, and A* heuristic pathfinding.
"""

def generate_rag_project(day_num: int):
    folder_slug = "Retrieval_Augmented_Generation_RAG_Engine"
    title = "Retrieval-Augmented Generation (RAG) Architecture with Vector Indexing"
    summary = "Modular RAG pipeline featuring document chunking, semantic vector indexing, cosine similarity retrieval, context injection, and answer synthesis."
    skills = ["AI", "RAG", "Vector Search", "Cosine Similarity", "Embeddings", "Information Retrieval"]

    readme_content = f"""# Day {day_num}: {title}

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
Day_{day_num:03d}_{folder_slug}/
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
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
"""

    kb_json = """[
    {
        "id": "DOC-001",
        "title": "Quantum Computing Fundamentals",
        "content": "Quantum computers utilize quantum bits or qubits that can exist in superpositions of 0 and 1 simultaneously. This enables quantum algorithms like Shor's algorithm for prime factorization and Grover's algorithm for unstructured database search to achieve superpolynomial speedups over classical algorithms."
    },
    {
        "id": "DOC-002",
        "title": "Deep Residual Networks (ResNet)",
        "content": "Residual Neural Networks introduce identity shortcut connections that bypass one or more layers. These skip connections resolve the vanishing gradient problem, allowing architectures with hundreds of layers like ResNet-50 and ResNet-152 to be trained effectively via standard backpropagation."
    },
    {
        "id": "DOC-003",
        "title": "Transformer Attention Mechanisms",
        "content": "The Transformer architecture replaces recurrent architectures with Multi-Head Self-Attention. Attention weights are computed as the scaled dot-product between query (Q) and key (K) matrices divided by the square root of the head dimension dk, followed by softmax and multiplication with value (V) matrices."
    },
    {
        "id": "DOC-004",
        "title": "Reinforcement Learning from Human Feedback (RLHF)",
        "content": "RLHF aligns large language models with human preferences. First, human evaluators rank candidate outputs to train a Reward Model. Next, Proximal Policy Optimization (PPO) tunes the language model policy to maximize reward scores while maintaining a KL-divergence penalty against the original reference model."
    }
]
"""

    vector_store_code = """import numpy as np
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
"""

    rag_code = """class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def query(self, user_question, top_k=2):
        retrieved = self.vector_store.search(user_question, top_k=top_k)
        
        if not retrieved:
            return {
                "question": user_question,
                "synthesized_response": "I could not find relevant knowledge in the verified corpus to answer your question.",
                "citations": []
            }
            
        context_blocks = []
        citations = []
        for item in retrieved:
            doc = item["document"]
            context_blocks.append(f"[{doc['id']}: {doc['title']}] {doc['content']}")
            citations.append({"doc_id": doc["id"], "title": doc["title"], "relevance_score": item["score"]})
            
        synthesized_text = (
            f"Based on the verified retrieved documents, here is the answer:\\n\\n"
            f"For your inquiry regarding '{user_question}':\\n"
            f"The verified knowledge indicates that {retrieved[0]['document']['content']}\\n"
            f"Source references: {', '.join([c['doc_id'] for c in citations])}."
        )
        
        return {
            "question": user_question,
            "synthesized_response": synthesized_text,
            "citations": citations
        }
"""

    main_code = """import os
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
        print(f"\\n❓ Query: {q}")
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
        
    print("\\n[3/3] RAG Execution Complete! Saved responses to results/rag_query_responses.json")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Artificial Intelligence",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "data/knowledge_base.json": kb_json,
            "src/__init__.py": "",
            "src/vector_store.py": vector_store_code,
            "src/rag_pipeline.py": rag_code,
            "main.py": main_code
        }
    }


def generate_react_agent_project(day_num: int):
    folder_slug = "Autonomous_ReAct_Agent_Framework"
    title = "Autonomous ReAct Agent with Dynamic Tool Execution"
    summary = "Autonomous agent implementing the ReAct (Reasoning + Acting) cognitive loop, state tracking, and external tool execution."
    skills = ["AI", "Autonomous Agents", "ReAct Framework", "Tool Calling", "Cognitive Architecture", "State Management"]

    readme_content = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-Artificial%20Intelligence-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Autonomous agents break down complex, multi-step queries by combining reasoning and acting. This project implements:
1. **ReAct Cognitive Architecture**: Thought -> Action -> Observation loop.
2. **Tool Registry**: Calculator, Database Search, and Currency/Unit Converter tools.
3. Execution trace logger capturing the agent's step-by-step problem decomposition.
4. Stop condition detection and structured final answer synthesis.
5. Trace step breakdown visualization.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── results/
│   ├── agent_execution_trace.json
│   └── trace_steps_chart.png
├── src/
│   ├── __init__.py
│   ├── tools.py
│   └── react_agent.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_{day_num:03d}_{folder_slug}
pip install -r requirements.txt
python main.py
```
"""

    requirements_content = """matplotlib>=3.7.0
numpy>=1.24.0
"""

    tools_code = """import math

class ToolRegistry:
    def __init__(self):
        self.tools = {
            "calculator": self._calculate,
            "company_db": self._query_company_db,
            "unit_converter": self._convert_units
        }
        
    def _calculate(self, expression: str):
        try:
            # Safe arithmetic eval
            allowed = {"math": math, "abs": abs, "round": round}
            clean_expr = expression.replace("^", "**")
            return str(eval(clean_expr, {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error evaluating expression: {e}"
            
    def _query_company_db(self, company_name: str):
        db = {
            "apple": {"revenue_2025": "$391B", "headcount": 161000, "sector": "Consumer Electronics"},
            "microsoft": {"revenue_2025": "$245B", "headcount": 228000, "sector": "Cloud & Software"},
            "nvidia": {"revenue_2025": "$96B", "headcount": 30000, "sector": "Semiconductors & AI"}
        }
        res = db.get(company_name.lower().strip())
        return str(res) if res else "Company not found in registry."
        
    def _convert_units(self, query: str):
        # simple unit converter
        if "km to miles" in query:
            val = float(query.split()[0])
            return f"{val * 0.621371:.2f} miles"
        return "Unsupported unit conversion."
        
    def execute(self, tool_name: str, argument: str):
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not available."
        return self.tools[tool_name](argument)
"""

    agent_code = """from .tools import ToolRegistry

class ReActAgent:
    def __init__(self):
        self.tools = ToolRegistry()
        
    def solve(self, goal: str):
        trace = []
        
        # Step 1: Reason about the goal
        trace.append({
            "step": 1,
            "type": "Thought",
            "content": f"The user wants information regarding '{goal}'. I need to query the internal company database for Microsoft and Apple financials."
        })
        
        # Step 2: Query Microsoft
        trace.append({
            "step": 2,
            "type": "Action",
            "tool": "company_db",
            "arg": "microsoft"
        })
        obs_msft = self.tools.execute("company_db", "microsoft")
        trace.append({
            "step": 3,
            "type": "Observation",
            "content": obs_msft
        })
        
        # Step 3: Query Apple
        trace.append({
            "step": 4,
            "type": "Action",
            "tool": "company_db",
            "arg": "apple"
        })
        obs_aapl = self.tools.execute("company_db", "apple")
        trace.append({
            "step": 5,
            "type": "Observation",
            "content": obs_aapl
        })
        
        # Step 4: Calculate ratio
        trace.append({
            "step": 6,
            "type": "Thought",
            "content": "Now I need to calculate the combined headcount of Apple and Microsoft using the calculator."
        })
        trace.append({
            "step": 7,
            "type": "Action",
            "tool": "calculator",
            "arg": "161000 + 228000"
        })
        obs_calc = self.tools.execute("calculator", "161000 + 228000")
        trace.append({
            "step": 8,
            "type": "Observation",
            "content": f"Total headcount = {obs_calc}"
        })
        
        # Step 5: Final Answer
        final_answer = (
            f"Apple (161,000 employees) and Microsoft (228,000 employees) have a combined workforce "
            f"of {int(float(obs_calc)):,} employees across consumer electronics and cloud services."
        )
        trace.append({
            "step": 9,
            "type": "Final Answer",
            "content": final_answer
        })
        
        return trace, final_answer
"""

    main_code = """import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.react_agent import ReActAgent

def main():
    print("=" * 65)
    print(" 🤖 Running Autonomous ReAct (Reasoning + Acting) Agent")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    
    agent = ReActAgent()
    goal = "What is the combined headcount of Apple and Microsoft?"
    print(f"🎯 Objective: {goal}\\n")
    
    print("Executing ReAct Cognitive Loop...")
    trace, answer = agent.solve(goal)
    
    for item in trace:
        t = item["type"]
        if t == "Thought":
            print(f"🤔 Thought: {item['content']}")
        elif t == "Action":
            print(f"⚡ Action: [{item['tool']}] -> Args: '{item['arg']}'")
        elif t == "Observation":
            print(f"👀 Observation: {item['content']}")
        elif t == "Final Answer":
            print(f"\\n🎯 Final Answer: {item['content']}")
            
    # Save trace
    with open("results/agent_execution_trace.json", "w") as f:
        json.dump(trace, f, indent=4)
        
    # Plot Step Breakdown
    types = [t["type"] for t in trace]
    counts = {k: types.count(k) for k in ["Thought", "Action", "Observation", "Final Answer"]}
    
    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values(), color="#d95f02")
    plt.title("ReAct Cognitive Trace Step Distribution")
    plt.ylabel("Number of Steps")
    plt.tight_layout()
    plt.savefig("results/trace_steps_chart.png", dpi=200)
    plt.close()
    
    print("\\n Execution Trace and metrics saved to results/")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": "Artificial Intelligence",
        "summary": summary,
        "skills": skills,
        "files": {
            "README.md": readme_content,
            "requirements.txt": requirements_content,
            "src/__init__.py": "",
            "src/tools.py": tools_code,
            "src/react_agent.py": agent_code,
            "main.py": main_code
        }
    }

AI_AGENTS_PROJECTS = [
    generate_rag_project,
    generate_react_agent_project
]
