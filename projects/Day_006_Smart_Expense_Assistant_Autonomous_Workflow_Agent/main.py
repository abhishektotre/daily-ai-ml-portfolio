import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.transaction_generator import create_expenses
from src.expense_agent import ExpenseAssistantAgent

def main():
    print("=" * 65)
    print(" 💳 Running Smart Expense Assistant & Autonomous Workflow Agent")
    print("=" * 65)
    
    print("[1/3] Generating corporate & personal transaction records...")
    df = create_expenses()
    print(f"      Ingested {len(df)} transactions.")
    
    print("[2/3] Executing autonomous agent audit & budget enforcement...")
    agent = ExpenseAssistantAgent()
    report = agent.process_and_audit(df)
    
    print("[3/3] Agent Workflow Audit Complete!")
    print(f"      - Total Expenditure: ${report['total_spend_usd']:,.2f}")
    print(f"      - Anomalies Flagged: {report['anomalies_flagged_count']}")
    print("      - Category Actions:")
    for a in report["category_audit_decisions"][:4]:
        print(f"        * [{a['category']}] Status: {a['status']} ({a['utilization_pct']}%) -> {a['recommendation']}")
    print("=" * 65)

if __name__ == "__main__":
    main()
