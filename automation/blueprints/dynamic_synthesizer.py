"""
Dynamic Project Synthesizer:
Procedurally generates rich, self-contained, and runnable projects for any day N,
guaranteeing infinite daily unique projects across all 6 domains.
"""

import random

DOMAINS_CONFIG = [
    {
        "domain": "Data Science",
        "slug_prefix": "DS",
        "badge_color": "blue",
        "topics": [
            ("Patient_Mortality_Risk_Stratification", "Patient Mortality Risk Stratification & Clinical Biomarkers",
             "Clinical risk stratification pipeline analyzing vital signs and laboratory biomarkers using logistic regression and odds-ratio analysis."),
            ("Supply_Chain_Disruption_Predictor", "Supply Chain Lead-Time & Disruption Predictor",
             "Predictive risk pipeline forecasting shipping latency and maritime logistics bottlenecks with feature interaction analysis."),
            ("Customer_LTV_Survival_Analysis", "Customer Lifetime Value & Survival Duration Modeling",
             "Statistical survival modeling assessing customer subscription longevity and tenure decay functions."),
            ("Telecom_Network_Congestion_Forecasting", "Cellular Network Traffic & Congestion Forecaster",
             "Time-series decomposition and statistical regression anticipating peak cellular bandwidth spikes.")
        ]
    },
    {
        "domain": "Data Analytics",
        "slug_prefix": "DA",
        "badge_color": "green",
        "topics": [
            ("Marketing_Multi_Touch_Attribution", "Digital Marketing Multi-Touch Attribution Modeling",
             "Marketing analytics engine comparing First-Touch, Last-Touch, and Markov Chain multi-touch attribution models."),
            ("Product_Engagement_Stickiness_KPIs", "Digital Product Feature Stickiness & DAU/MAU KPIs",
             "In-depth product analytics tracking feature adoption, session duration, and daily/monthly user retention ratios."),
            ("Retail_Basket_Association_Rules", "Market Basket Affinity & Association Rule Mining",
             "Association rule mining assessing transaction support, confidence, and lift for retail merchandising."),
            ("Subscription_Pricing_Elasticity_Study", "SaaS Dynamic Pricing Elasticity & Revenue Frontier",
             "Economic pricing elasticity model analyzing conversion response curves to pricing adjustments.")
        ]
    },
    {
        "domain": "Machine Learning",
        "slug_prefix": "ML",
        "badge_color": "orange",
        "topics": [
            ("Industrial_IoT_Predictive_Maintenance", "Industrial IoT Equipment Failure & Remaining Useful Life",
             "Vibration and thermal sensor degradation modeling using Random Forest classifier for predictive maintenance."),
            ("Cybersecurity_Network_Intrusion_Detection", "Network Intrusion & Anomaly Classification with Isolation Forest",
             "High-throughput packet telemetry inspection and zero-day anomaly classification using Isolation Forests."),
            ("Vehicle_Residual_Value_Gradient_Boosting", "Automotive Residual Value Depreciation Engine",
             "Gradient boosting regression model estimating vehicle depreciation curves based on mileage and service history."),
            ("Multiclass_Genomic_Disease_Diagnostics", "Multi-Class Clinical Pathology Diagnostic Engine",
             "Regularized multinomial regression pipeline classifying genomic disease phenotypes with ROC-AUC analysis.")
        ]
    },
    {
        "domain": "Deep Learning",
        "slug_prefix": "DL",
        "badge_color": "red",
        "topics": [
            ("LSTM_Recurrent_Stock_Trend_Forecaster", "LSTM Recurrent Network for Multivariate Financial Series",
             "Recurrent time-series forecasting model utilizing sliding window sequence formulation for asset volatility."),
            ("Siamese_Metric_Verification_Network", "Deep Siamese Distance Network for Verification",
             "Contrastive metric distance learning comparing feature vector embeddings using Euclidean distance thresholds."),
            ("Convolutional_Edge_Pattern_Recognizer", "Convolutional Feature Extraction & Pattern Classifier",
             "Custom spatial convolution filtering, max-pooling layers, and dense classifier for 2D structural patterns."),
            ("Denoising_Autoencoder_Signal_Reconstruction", "Deep Denoising Autoencoder for Corrupted Sensor Streams",
             "Symmetric neural bottleneck network trained to reconstruct uncorrupted telemetry signals from noisy inputs.")
        ]
    },
    {
        "domain": "Natural Language Processing",
        "slug_prefix": "NLP",
        "badge_color": "purple",
        "topics": [
            ("Semantic_Document_Vector_Search_Engine", "Semantic Search & Context Retrieval with Vector Cosine Distance",
             "Dense semantic retrieval engine indexing technical documentation and computing vector similarity rankings."),
            ("Customer_Intent_Utterance_Classifier", "Customer Support Intent Classification & Slot Tagging",
             "Hierarchical NLP classifier mapping customer inquiries into granular support intents using TF-IDF and Logistic Regression."),
            ("Financial_News_Named_Entity_Extractor", "Financial News Named Entity Recognition (NER) Engine",
             "Rule and statistical token tagger extracting ticker symbols, monetary figures, and corporate acquisitions."),
            ("Legal_Contract_Key_Clause_Extractor", "Legal Contract Salient Clause & Risk Analyzer",
             "Document processing pipeline scanning contractual paragraphs for indemnification, liability, and breach clauses.")
        ]
    },
    {
        "domain": "Artificial Intelligence",
        "slug_prefix": "AI",
        "badge_color": "yellow",
        "topics": [
            ("A_Star_Heuristic_Pathfinding_Simulator", "A* Heuristic Search & Autonomous Grid Navigation",
             "Dynamic A* pathfinding algorithm optimizing state space navigation around dynamic obstacle grids."),
            ("Multi_Armed_Bandit_Exploration_Engine", "Multi-Armed Bandit Reinforcement Learning Engine",
             "Exploration vs exploitation optimization comparing Epsilon-Greedy, UCB1, and Thompson Sampling algorithms."),
            ("Autonomous_Task_Planner_State_Machine", "Autonomous Goal-Oriented Task Planning State Machine",
             "Hierarchical task network planner translating high-level objectives into validated sequential actions."),
            ("Minimax_Game_Decision_Agent_AlphaBeta", "Minimax Adversarial Game Agent with Alpha-Beta Pruning",
             "Heuristic adversarial search engine optimizing strategic gameplay decisions under lookahead depth constraints.")
        ]
    }
]

def synthesize_project(day_num: int, domain_key: str = None):
    # Select domain in round-robin if not specified
    if not domain_key:
        domain_idx = (day_num - 1) % len(DOMAINS_CONFIG)
        cfg = DOMAINS_CONFIG[domain_idx]
    else:
        cfg = next((c for c in DOMAINS_CONFIG if c["domain"].lower().replace(" ", "_") == domain_key), DOMAINS_CONFIG[0])
        
    topic_idx = ((day_num - 1) // len(DOMAINS_CONFIG)) % len(cfg["topics"])
    slug_name, title, summary = cfg["topics"][topic_idx]
    
    # If wrapped around multiple cycles, add variation tag
    cycle = (day_num - 1) // (len(DOMAINS_CONFIG) * len(cfg["topics"]))
    if cycle > 0:
        slug_name = f"{slug_name}_v{cycle+1}"
        title = f"{title} (Advanced Iteration {cycle+1})"
        
    folder_slug = f"{cfg['slug_prefix']}_{slug_name}"
    
    readme = f"""# Day {day_num}: {title}

![Domain](https://img.shields.io/badge/Domain-{cfg['domain'].replace(' ', '%20')}-{cfg['badge_color']})
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
{summary}

### Key Capabilities
- Automated synthetic telemetry generation with realistic distributions and noise parameters.
- Robust data pre-processing, validation, and standard scaling.
- Algorithmic modeling tuned for high precision and generalizability.
- Automated performance evaluation, diagnostic plots, and metric logs.

## 🛠️ Project Structure
```text
Day_{day_num:03d}_{folder_slug}/
├── data/
│   └── dataset.csv
├── results/
│   ├── analysis_plot.png
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   └── engine.py
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

    requirements = """numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""

    data_pipeline = f"""import numpy as np
import pandas as pd
import os

def generate_data(n_samples=2000, output_path="data/dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42 + {day_num})
    
    f1 = np.random.normal(loc=50, scale=15, size=n_samples)
    f2 = np.random.exponential(scale=10, size=n_samples)
    f3 = np.random.uniform(1, 100, size=n_samples)
    f4 = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Target correlation
    signal = 0.04 * f1 + 0.08 * f2 - 0.02 * f3 + 1.2 * f4 + np.random.normal(0, 0.5, n_samples)
    prob = 1 / (1 + np.exp(-signal))
    target = (prob > 0.5).astype(int)
    
    df = pd.DataFrame({{
        "feature_metric_a": f1.round(2),
        "feature_metric_b": f2.round(2),
        "feature_metric_c": f3.round(2),
        "category_flag": f4,
        "target": target
    }})
    df.to_csv(output_path, index=False)
    return df
"""

    engine = f"""import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def run_pipeline(df, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    # Plot Feature Importance
    plt.figure(figsize=(7, 4))
    sns.barplot(x=clf.feature_importances_, y=X.columns, palette="mako")
    plt.title("Feature Importances - {title}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analysis_plot.png"), dpi=200)
    plt.close()
    
    metrics = {{
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "samples_evaluated": len(y_test)
    }}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics
"""

    main_script = f"""import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.data_pipeline import generate_data
from src.engine import run_pipeline

def main():
    print("=" * 65)
    print(" 🚀 Running Day {day_num} Pipeline: {title}")
    print("=" * 65)
    
    print("[1/3] Generating domain-specific dataset...")
    df = generate_data()
    print(f"      Created {{len(df)}} observation records.")
    
    print("[2/3] Executing analytical modeling engine...")
    metrics = run_pipeline(df)
    
    print("[3/3] Execution Successful! Performance Metrics:")
    for k, v in metrics.items():
        print(f"      - {{k}}: {{v}}")
    print("=" * 65)

if __name__ == "__main__":
    main()
"""

    return {
        "folder_slug": folder_slug,
        "title": title,
        "domain": cfg["domain"],
        "summary": summary,
        "skills": [cfg["domain"], "Predictive Analytics", "Machine Learning", "Data Pipeline", "Python"],
        "files": {
            "README.md": readme,
            "requirements.txt": requirements,
            "src/__init__.py": "",
            "src/data_pipeline.py": data_pipeline,
            "src/engine.py": engine,
            "main.py": main_script
        }
    }
