"""
Daily Project Generation Engine:
Orchestrates domain rotation, project synthesis, file generation,
execution, state tracking, and portfolio catalog updates.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .blueprints import DOMAIN_REGISTRY, DOMAIN_ORDER
from .blueprints.dynamic_synthesizer import synthesize_project
from .update_readme import update_master_readme

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "current_day": 0,
            "total_projects": 0,
            "last_run_date": None,
            "domain_counts": {
                "Data Science": 0,
                "Data Analytics": 0,
                "Machine Learning": 0,
                "Deep Learning": 0,
                "Natural Language Processing": 0,
                "Artificial Intelligence": 0
            },
            "projects_history": []
        }
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def get_used_signatures(state):
    used = set()
    # From history in state
    for p in state.get("projects_history", []):
        f = p.get("folder", "")
        slug = f.split("_", 2)[-1] if "_" in f else f
        used.add(slug.lower().strip())
        used.add(p.get("title", "").lower().strip())

    # Also scan physical projects directory
    if os.path.exists(PROJECTS_DIR):
        for entry in os.listdir(PROJECTS_DIR):
            parts = entry.split("_", 2)
            if len(parts) >= 3:
                used.add(parts[2].lower().strip())
    return used

def generate_next_project(target_day: int = None, domain_override: str = None, execute: bool = True, deploy: bool = False, force: bool = False):
    state = load_state()

    # Enforce strict 1-project-per-calendar-day cadence
    today_str = datetime.now().strftime("%Y-%m-%d")
    last_run_str = state.get("last_run_date", "")[:10] if state.get("last_run_date") else ""
    if last_run_str == today_str and not force and target_day is None:
        print("=" * 65)
        print(f"🛑 Daily Lock Active: Today's project (Day {state.get('current_day', 0)}) was already deployed on {today_str}.")
        print("   To preserve a strict 1-project-per-day cadence on GitHub,")
        print(f"   Day {state.get('current_day', 0) + 1} will unlock tomorrow.")
        print("   (Use '--force' if you intentionally wish to override this lock).")
        print("=" * 65)
        return None

    next_day = target_day if target_day is not None else state["current_day"] + 1

    # Domain selection
    if domain_override and domain_override in DOMAIN_REGISTRY:
        domain_key = domain_override
    else:
        domain_idx = (next_day - 1) % len(DOMAIN_ORDER)
        domain_key = DOMAIN_ORDER[domain_idx]

    domain_info = DOMAIN_REGISTRY[domain_key]
    domain_name = domain_info["name"]
    blueprint_list = domain_info["projects"]

    used_sigs = get_used_signatures(state)

    # 1. Select first available blueprint in this domain that has not been used
    project = None
    for bp_func in blueprint_list:
        candidate = bp_func(next_day)
        cand_slug = candidate["folder_slug"].lower().strip()
        cand_title = candidate["title"].lower().strip()
        if cand_slug not in used_sigs and cand_title not in used_sigs:
            project = candidate
            break

    # 2. If all curated blueprints for this domain have been used, dynamically synthesize a unique project
    if project is None:
        project = synthesize_project(next_day, domain_key, used_slugs=used_sigs)

    folder_name = f"Day_{next_day:03d}_{project['folder_slug']}"
    target_project_dir = os.path.join(PROJECTS_DIR, folder_name)
    os.makedirs(target_project_dir, exist_ok=True)

    print(f"✨ Generating Day {next_day:03d}: {project['title']} [{domain_name}]")
    print(f"   Destination: {target_project_dir}")

    # Write files
    for rel_path, content in project["files"].items():
        file_path = os.path.join(target_project_dir, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"   Created {len(project['files'])} project files.")

    # Execute project if requested
    if execute:
        print(f"⚡ Executing project pipeline (python main.py)...")
        main_script = os.path.join(target_project_dir, "main.py")
        if os.path.exists(main_script):
            original_cwd = os.getcwd()
            original_sys_path = list(sys.path)
            try:
                os.chdir(target_project_dir)
                sys.path.insert(0, target_project_dir)
                import runpy
                runpy.run_path("main.py", run_name="__main__")
                print("   Pipeline execution completed successfully!")
            except Exception as e:
                print(f"   ⚠️ Pipeline execution warning/error: {e}")
            finally:
                os.chdir(original_cwd)
                sys.path = original_sys_path

    # Update state
    state["current_day"] = next_day
    state["total_projects"] += 1
    state["last_run_date"] = datetime.now().isoformat()
    state["domain_counts"][domain_name] = state["domain_counts"].get(domain_name, 0) + 1
    state["projects_history"].append({
        "day": next_day,
        "title": project["title"],
        "domain": domain_name,
        "folder": f"projects/{folder_name}",
        "skills": project.get("skills", []),
        "created_at": datetime.now().isoformat()
    })
    save_state(state)

    # Refresh Master README
    update_master_readme()

    # Deploy as standalone repository if requested
    if deploy:
        from pathlib import Path
        from .deployer import deploy_project_as_repo
        deploy_project_as_repo(
            Path(target_project_dir),
            title=project["title"],
            summary=project.get("summary", "")
        )

    return {
        "day": next_day,
        "title": project["title"],
        "domain": domain_name,
        "folder": folder_name,
        "project_dir": target_project_dir
    }
