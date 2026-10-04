"""
Master Showcase README Updater:
Scans state.json and projects/ to generate a master portfolio README.md
complete with badges, progress stats, domain breakdowns, and project links.
"""

import json
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
README_FILE = os.path.join(ROOT_DIR, "README.md")

DOMAIN_BADGE_MAP = {
    "Data Science": "https://img.shields.io/badge/Data%20Science-blue",
    "Data Analytics": "https://img.shields.io/badge/Data%20Analytics-green",
    "Machine Learning": "https://img.shields.io/badge/Machine%20Learning-orange",
    "Deep Learning": "https://img.shields.io/badge/Deep%20Learning-red",
    "Natural Language Processing": "https://img.shields.io/badge/NLP-purple",
    "Artificial Intelligence": "https://img.shields.io/badge/AI%20%26%20Agents-yellow"
}

def update_master_readme():
    if not os.path.exists(STATE_FILE):
        state = {
            "current_day": 0,
            "total_projects": 0,
            "last_run_date": None,
            "domain_counts": {},
            "projects_history": []
        }
    else:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

    total_projects = state.get("total_projects", 0)
    current_day = state.get("current_day", 0)
    domain_counts = state.get("domain_counts", {})
    history = state.get("projects_history", [])
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Table of projects
    if not history:
        table_rows = "| - | - | *Automation initializing... First project will appear here shortly!* | - | - |"
    else:
        rows = []
        for p in reversed(history):
            day_fmt = f"**Day {p['day']:03d}**"
            domain = p.get("domain", "Data Science")
            badge = f"![{domain}]({DOMAIN_BADGE_MAP.get(domain, 'https://img.shields.io/badge/Field-blue')})"
            title = p.get("title", "Project")
            target_url = p.get("repo_url") or folder
            link = f"[{title}]({target_url})" if target_url else title
            skills = ", ".join([f"`{s}`" for s in p.get("skills", [])[:3]])
            rows.append(f"| {day_fmt} | {badge} | {link} | {skills} | ✅ Completed |")
        table_rows = "\n".join(rows)

    # Domain summary table
    domain_summary_rows = []
    for d, badge_url in DOMAIN_BADGE_MAP.items():
        count = domain_counts.get(d, 0)
        domain_summary_rows.append(f"| ![{d}]({badge_url}) | **{count}** projects |")
    domain_summary_table = "\n".join(domain_summary_rows)

    readme_content = f"""# 🧠 Daily AI, ML, Data Science & Analytics Portfolio

[![Total Projects](https://img.shields.io/badge/Total%20Projects-{total_projects}-blueviolet?style=for-the-badge&logo=github)](projects/)
[![Current Streak](https://img.shields.io/badge/Daily%20Streak-{current_day}%20Days-orange?style=for-the-badge&logo=fire)](projects/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Daily Automation](https://img.shields.io/badge/Automation-Active%20%28GitHub%20Actions%29-brightgreen?style=for-the-badge&logo=githubactions)](.github/workflows/daily-project.yml)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

> **Autonomous Daily Repository**: Every single day, an automated pipeline designs, tests, builds, and commits a production-ready, self-contained project spanning **Data Science**, **Data Analytics**, **Machine Learning**, **Deep Learning**, **NLP**, and **Artificial Intelligence**.

---

## 📊 Domain Distribution Matrix

{domain_summary_table}

---

## 📂 Daily Project Showcase Directory

| Day | Domain | Project & Architecture | Core Tools / Techniques | Status |
| :---: | :---: | :--- | :--- | :---: |
{table_rows}

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

*Last synchronized: `{last_updated}`*
"""

    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✅ Successfully updated {README_FILE} (Total projects: {total_projects})")

if __name__ == "__main__":
    update_master_readme()
