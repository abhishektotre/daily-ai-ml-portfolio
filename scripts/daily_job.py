"""
Headless, Non-Interactive Daily Runner for Task Scheduler and Background Daemons.
Executes the daily project pipeline, captures logs, and prevents interactive hanging.
"""

import sys
import os
import datetime
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "daily_automation.log"

def sync_remote():
    try:
        from automation.config import CONFIG
        token = CONFIG.get("GITHUB_TOKEN")
        username = CONFIG.get("GITHUB_USERNAME")
        if token and username:
            auth_url = f"https://{username}:{token}@github.com/{username}/daily-ai-ml-portfolio.git"
            subprocess.run(["git", "pull", auth_url, "main"], cwd=str(BASE_DIR), capture_output=True, timeout=30)
    except Exception:
        pass

def run_job():
    sync_remote()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"⏰ [AUTOMATION TRIGGER] {now_str}\n")
        f.write(f"{'='*70}\n")
        f.flush()

        python_exe = "C:\\Users\\abhis\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        cmd = [python_exe, "-m", "automation.runner", "--generate", "--execute", "--deploy"]
        
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        finish_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n🏁 [COMPLETED] Exit Code: {proc.returncode} at {finish_str}\n")
        f.write(f"{'='*70}\n")

if __name__ == "__main__":
    run_job()
