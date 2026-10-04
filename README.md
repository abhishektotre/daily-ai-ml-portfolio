# 🧠 Daily AI, ML, Data Science & Analytics Portfolio

[![Total Projects](https://img.shields.io/badge/Total%20Projects-2-blueviolet?style=for-the-badge&logo=github)](projects/)
[![Current Streak](https://img.shields.io/badge/Daily%20Streak-2%20Days-orange?style=for-the-badge&logo=fire)](projects/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Daily Automation](https://img.shields.io/badge/Automation-Active%20%28GitHub%20Actions%29-brightgreen?style=for-the-badge&logo=githubactions)](.github/workflows/daily-project.yml)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

> **Autonomous Daily Repository**: Every single day, an automated pipeline designs, tests, builds, and commits a production-ready, self-contained project spanning **Data Science**, **Data Analytics**, **Machine Learning**, **Deep Learning**, **NLP**, and **Artificial Intelligence**.

---

## 📊 Domain Distribution Matrix

| ![Data Science](https://img.shields.io/badge/Data%20Science-blue) | **1** projects |
| ![Data Analytics](https://img.shields.io/badge/Data%20Analytics-green) | **1** projects |
| ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-orange) | **0** projects |
| ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-red) | **0** projects |
| ![Natural Language Processing](https://img.shields.io/badge/NLP-purple) | **0** projects |
| ![Artificial Intelligence](https://img.shields.io/badge/AI%20%26%20Agents-yellow) | **0** projects |

---

## 📂 Daily Project Showcase Directory

| Day | Domain | Project & Architecture | Core Tools / Techniques | Status |
| :---: | :---: | :--- | :--- | :---: |
| **Day 002** | ![Data Analytics](https://img.shields.io/badge/Data%20Analytics-green) | [SaaS Monthly Cohort Retention & Churn Analytics](https://github.com/abhishektotre/saas-cohort-retention-analytics) | `Data Analytics`, `Cohort Analysis`, `Retention Matrix` | ✅ Completed |
| **Day 001** | ![Data Science](https://img.shields.io/badge/Data%20Science-blue) | [Customer Churn Prediction & Retention Analytics](https://github.com/abhishektotre/customer-churn-prediction) | `Data Science`, `EDA`, `Feature Engineering` | ✅ Completed |

---

## 🏗️ Architecture & Automated Execution Flow

The system runs autonomously via **GitHub Actions** (cloud cron) and can also be triggered **locally** via CLI:

```mermaid
flowchart LR
    A["⏰ GitHub Actions Cron (00:00 UTC)"] --> B["⚙️ Automation Engine (runner.py)"]
    L["💻 Local CLI / Task Scheduler"] --> B
    B --> C["🎯 Domain Selector (Round-Robin)"]
    C --> D["🛠️ Blueprint & Synthesizer"]
    D --> E["📁 Create Project Directory"]
    E --> F["🧪 Execute Pipeline (python main.py)"]
    F --> G["📈 Generate Results & Plots"]
    G --> H["📝 Update Master README.md & state.json"]
    H --> I["🚀 Auto Git Commit & Push to GitHub"]
```

Each daily project folder is **100% self-contained** and includes:
- `README.md` (Detailed problem statement, architectural diagram, metrics, and instructions)
- `main.py` (End-to-end runnable entrypoint)
- `src/` (Clean, modular code: data generators, model trainers, evaluation engines)
- `data/` (Synthetic realistic datasets generated on-the-fly)
- `results/` (Generated metric scorecards, confusion matrices, and EDA plots)
- `requirements.txt` (Pinned dependencies)

---

## 🚀 Running Any Project Locally

To run any daily project on your local machine:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# 2. Enter any day's project folder
cd projects/Day_001_Customer_Churn_Prediction

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Run the end-to-end pipeline
python main.py
```

---

## ⚙️ Triggering the Automation Manually

You can trigger the generator anytime locally:

```bash
# Generate and execute the next daily project
python automation/runner.py --generate --execute

# Generate, execute, and push immediately to GitHub
python automation/runner.py --generate --execute --push

# Check portfolio status and stats
python automation/runner.py --status
```

*Last synchronized: `2026-10-04 18:28:40 UTC`*
