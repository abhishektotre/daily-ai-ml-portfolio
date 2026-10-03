"""
Command-Line Interface (CLI) Runner for Daily Project Automation:
Handles project generation, execution, git staging, committing, and pushing.
"""

import argparse
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .generator import generate_next_project, load_state
from .blueprints import DOMAIN_ORDER, DOMAIN_REGISTRY

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_git_command(args, cwd=BASE_DIR):
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return result

def commit_and_push(project_info):
    day = project_info["day"]
    title = project_info["title"]
    domain = project_info["domain"]

    print("📦 Staging changes for Git...")
    run_git_command(["add", "."])

    commit_msg = f"🚀 Day {day:03d}: {title} [{domain}]"
    print(f"📝 Committing: {commit_msg}")
    res_commit = run_git_command(["commit", "-m", commit_msg])
    if res_commit.returncode != 0:
        print(f"Commit output: {res_commit.stdout or res_commit.stderr}")

    print("🚀 Pushing commits to remote...")
    res_push = run_git_command(["push", "origin", "main"])
    if res_push.returncode == 0:
        print("✅ Successfully pushed changes to GitHub!")
    else:
        print(f"⚠️ Push status (remote may not be configured yet or requires authentication):")
        print(res_push.stderr or res_push.stdout)

def show_status():
    state = load_state()
    current_day = state.get("current_day", 0)
    total_projects = state.get("total_projects", 0)
    domain_counts = state.get("domain_counts", {})

    next_day = current_day + 1
    next_domain_key = DOMAIN_ORDER[(next_day - 1) % len(DOMAIN_ORDER)]
    next_domain_name = DOMAIN_REGISTRY[next_domain_key]["name"]

    print("=" * 60)
    print(" 🌟 Daily AI / ML / Data Science Automation Status")
    print("=" * 60)
    print(f" • Current Day:         Day {current_day}")
    print(f" • Total Projects:      {total_projects}")
    print(f" • Last Run Date:       {state.get('last_run_date', 'Never')}")
    print(f" • Next Up (Day {next_day}):   {next_domain_name}")
    print("\n 📊 Domain Breakdown:")
    for domain, count in domain_counts.items():
        print(f"   - {domain:<30}: {count} projects")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Daily Project Automation Runner")
    parser.add_argument("--generate", action="store_true", help="Generate next daily project")
    parser.add_argument("--execute", action="store_true", help="Execute the generated project pipeline")
    parser.add_argument("--push", action="store_true", help="Commit and push changes to Git")
    parser.add_argument("--deploy", action="store_true", help="Deploy generated project to its own standalone GitHub repository")
    parser.add_argument("--deploy-all", action="store_true", help="Deploy all existing projects as standalone GitHub repositories")
    parser.add_argument("--force", action="store_true", help="Override the 1-project-per-calendar-day lock")
    parser.add_argument("--status", action="store_true", help="Display current status and domain breakdown")
    parser.add_argument("--day", type=int, default=None, help="Explicit target day number")
    parser.add_argument("--domain", type=str, default=None, choices=DOMAIN_ORDER, help="Override target domain")

    args = parser.parse_args()

    # If no flags passed, default to showing status or executing generation
    if not (args.generate or args.execute or args.push or args.status or args.deploy or args.deploy_all):
        parser.print_help()
        sys.exit(0)

    if args.deploy_all:
        from .deployer import deploy_all_projects
        deploy_all_projects()
        return

    if args.status:
        show_status()
        return

    if args.generate:
        execute_pipeline = args.execute
        info = generate_next_project(
            target_day=args.day,
            domain_override=args.domain,
            execute=execute_pipeline,
            deploy=args.deploy,
            force=args.force
        )
        if info:
            print(f"\n🎉 Successfully created Day {info['day']:03d} project at {info['project_dir']}")
            if args.push:
                commit_and_push(info)

if __name__ == "__main__":
    main()
