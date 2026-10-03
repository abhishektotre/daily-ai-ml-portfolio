"""
Standalone Repository Deployer:
Automatically initializes an independent Git repository for each daily project,
creates a remote repository on GitHub via REST API, and pushes the project code.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import requests

from .config import load_env

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / "projects"

def format_repo_name(slug_or_folder: str, prefix: str = "") -> str:
    # Remove leading Day_XXX_ if present
    clean_name = re.sub(r"^Day_\d+_", "", slug_or_folder)
    # Convert underscores and spaces to hyphens, lowercase
    clean_name = re.sub(r"[_\s]+", "-", clean_name).lower().strip("-")
    if prefix:
        return f"{prefix.strip('-')}-{clean_name}"
    return clean_name

def create_github_repo(repo_name: str, description: str, token: str, username: str, private: bool = False):
    if not token:
        return False, "No GitHub Personal Access Token (GITHUB_TOKEN) provided."

    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    payload = {
        "name": repo_name,
        "description": description[:350],
        "private": private,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": False,
        "auto_init": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 201:
            return True, f"Created repository https://github.com/{username}/{repo_name}"
        elif response.status_code == 422:
            # 422 usually indicates repo already exists on this account
            data = response.json()
            errors = [e.get("message", "") for e in data.get("errors", [])]
            if any("already exists" in e.lower() for e in errors) or "name already exists" in str(data).lower():
                return True, f"Repository https://github.com/{username}/{repo_name} already exists. Proceeding to push."
            return False, f"GitHub API error (422): {response.text}"
        elif response.status_code == 401:
            return False, "GitHub Authentication Failed (401). Check your GITHUB_TOKEN permissions."
        else:
            return False, f"GitHub API returned HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Network exception communicating with GitHub API: {e}"

def run_cmd(cmd_list, cwd):
    result = subprocess.run(cmd_list, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result

def deploy_project_as_repo(project_dir: Path, title: str = "", summary: str = "", config: dict = None):
    if config is None:
        config = load_env()

    username = config.get("GITHUB_USERNAME", "abhishektotre")
    token = config.get("GITHUB_TOKEN", "")
    prefix = config.get("REPO_PREFIX", "")
    is_private = config.get("REPO_VISIBILITY", "public").lower() == "private"

    folder_name = project_dir.name
    repo_name = format_repo_name(folder_name, prefix=prefix)
    description = summary or f"{title} - Autonomous Daily AI/ML Project"

    print("=" * 65)
    print(f" 🚀 Deploying Standalone Repository: {repo_name}")
    print(f" 📁 Local Directory: {project_dir}")
    print("=" * 65)

    # 1. Ensure .gitignore and LICENSE exist inside the project
    proj_gitignore = project_dir / ".gitignore"
    if not proj_gitignore.exists():
        root_gitignore = BASE_DIR / ".gitignore"
        if root_gitignore.exists():
            shutil.copy(root_gitignore, proj_gitignore)

    proj_license = project_dir / "LICENSE"
    if not proj_license.exists():
        root_license = BASE_DIR / "LICENSE"
        if root_license.exists():
            shutil.copy(root_license, proj_license)

    # 2. Initialize Git repository inside project directory
    git_dir = project_dir / ".git"
    if not git_dir.exists():
        print(" [1/4] Initializing independent Git repository...")
        res = run_cmd(["git", "init", "-b", "main"], cwd=project_dir)
        if res.returncode != 0:
            print(f"       Git init warning: {res.stderr or res.stdout}")
    else:
        print(" [1/4] Existing Git repository detected.")

    # Configure local git user if not present
    run_cmd(["git", "config", "user.name", "Abhishek"], cwd=project_dir)
    run_cmd(["git", "config", "user.email", "abhishektotre22@gmail.com"], cwd=project_dir)

    # 3. Stage & Commit
    print(" [2/4] Staging and committing project files...")
    run_cmd(["git", "add", "-A"], cwd=project_dir)
    commit_msg = f"🚀 Initial release: {title or repo_name}"
    res_commit = run_cmd(["git", "commit", "-m", commit_msg], cwd=project_dir)
    if "nothing to commit" in (res_commit.stdout or ""):
        print("       Working tree clean; no new changes to commit.")
    else:
        print(f"       Committed changes: '{commit_msg}'")

    # 4. Create Remote Repository on GitHub via API
    print(" [3/4] Ensuring GitHub remote repository exists...")
    api_success, api_msg = create_github_repo(repo_name, description, token, username, private=is_private)
    print(f"       {api_msg}")

    # 5. Configure Remote and Push
    print(" [4/4] Pushing to GitHub remote repository...")
    if token:
        remote_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    else:
        remote_url = f"https://github.com/{username}/{repo_name}.git"

    # Reset origin remote
    run_cmd(["git", "remote", "remove", "origin"], cwd=project_dir)
    res_remote = run_cmd(["git", "remote", "add", "origin", remote_url], cwd=project_dir)

    # Push to origin main
    res_push = run_cmd(["git", "push", "-u", "origin", "main", "--force"], cwd=project_dir)
    clean_url = f"https://github.com/{username}/{repo_name}"

    if res_push.returncode == 0:
        print(f"\n🎉 SUCCESS! Deployed to standalone repository:\n   👉 {clean_url}\n")
        return True, clean_url
    else:
        err_msg = (res_push.stderr or res_push.stdout) or ""
        print(f"\n⚠️ Push status for {clean_url}:")
        if not token:
            print("   👉 Note: No GITHUB_TOKEN configured.")
            print(f"      To allow automated creation of repo '{repo_name}' on your GitHub account,")
            print("      set GITHUB_TOKEN in your .env file or environment variables.")
        elif "Authentication failed" in err_msg or "could not read Username" in err_msg or "403" in err_msg:
            print("   👉 GitHub Authentication failed. Verify that your GITHUB_TOKEN has 'repo' permissions.")
        elif "Repository not found" in err_msg:
            print(f"   👉 Repository '{repo_name}' was not found on your GitHub account.")
            print("      Verify that your token has repository creation permissions.")
        else:
            print(f"   {err_msg if err_msg.strip() else 'Waiting for GitHub remote authentication / token configuration.'}")
        return False, clean_url

def deploy_all_projects():
    config = load_env()
    state_file = BASE_DIR / "automation" / "state.json"
    history_map = {}
    if state_file.exists():
        with open(state_file, "r") as f:
            state = json.load(f)
            for p in state.get("projects_history", []):
                folder = p.get("folder", "").replace("projects/", "").strip("/")
                history_map[folder] = p

    if not PROJECTS_DIR.exists():
        print("No projects directory found.")
        return

    subdirs = sorted([d for d in PROJECTS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} daily projects to deploy separately as individual GitHub repositories.")

    results = []
    for p_dir in subdirs:
        meta = history_map.get(p_dir.name, {})
        title = meta.get("title", p_dir.name)
        skills = meta.get("skills", [])
        summary = f"{title} - Autonomous Daily Data Science & AI Project ({', '.join(skills[:3])})"
        ok, url = deploy_project_as_repo(p_dir, title=title, summary=summary, config=config)
        results.append((p_dir.name, ok, url))

    print("\n" + "=" * 65)
    print(" 📋 Standalone Repositories Deployment Summary")
    print("=" * 65)
    for name, ok, url in results:
        status_icon = "✅" if ok else "⚠️"
        print(f" {status_icon} {name:<45} -> {url}")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        deploy_all_projects()
    elif len(sys.argv) > 1:
        target = Path(sys.argv[1])
        deploy_project_as_repo(target)
    else:
        deploy_all_projects()
