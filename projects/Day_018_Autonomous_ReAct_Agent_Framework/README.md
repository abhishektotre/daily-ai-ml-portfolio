# Day 18: Autonomous ReAct Agent with Dynamic Tool Execution

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
Day_018_Autonomous_ReAct_Agent_Framework/
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
cd Day_018_Autonomous_ReAct_Agent_Framework
pip install -r requirements.txt
python main.py
```
