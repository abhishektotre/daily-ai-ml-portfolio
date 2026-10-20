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
from src.react_agent import ReActAgent

def main():
    print("=" * 65)
    print(" 🤖 Running Autonomous ReAct (Reasoning + Acting) Agent")
    print("=" * 65)
    
    os.makedirs("results", exist_ok=True)
    
    agent = ReActAgent()
    goal = "What is the combined headcount of Apple and Microsoft?"
    print(f"🎯 Objective: {goal}\n")
    
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
            print(f"\n🎯 Final Answer: {item['content']}")
            
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
    
    print("\n Execution Trace and metrics saved to results/")
    print("=" * 65)

if __name__ == "__main__":
    main()
