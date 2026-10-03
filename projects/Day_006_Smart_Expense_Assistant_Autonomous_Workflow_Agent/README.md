# Day 6: Smart Expense Assistant with Autonomous Agent Workflows

![Domain](https://img.shields.io/badge/Domain-Artificial%20Intelligence-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Managing personal and corporate expenditures requires automated categorization, receipt metadata extraction, and policy enforcement. Inspired by intelligent expense assistants, this project implements:
1. Multi-stream expense transaction ingestion (merchants, amounts, payment methods, transaction timestamps).
2. NLP & Rule-based contextual categorization engine (Dining, Utilities, Subscriptions, Travel, Healthcare).
3. Statistical outlier and abnormal spending burst detection.
4. Autonomous workflow execution: automated expense approval, policy violations flagging, and budget rebalancing advice.
5. Visual expense category breakdown and spending timeline alerts.

## 🛠️ Project Structure
```text
Day_006_Smart_Expense_Assistant_Autonomous_Workflow_Agent/
├── data/
│   └── expense_transactions.csv
├── results/
│   ├── category_spend_pie.png
│   ├── spending_timeline.png
│   └── agent_workflow_audit.json
├── src/
│   ├── __init__.py
│   ├── transaction_generator.py
│   └── expense_agent.py
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 How to Run
```bash
cd Day_006_Smart_Expense_Assistant_Autonomous_Workflow_Agent
pip install -r requirements.txt
python main.py
```
