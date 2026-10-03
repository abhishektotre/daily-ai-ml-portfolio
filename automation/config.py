"""
Configuration settings for GitHub automation & standalone repository deployment.
Reads from environment variables or .env file.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

def load_env():
    config = {
        "GITHUB_USERNAME": os.environ.get("GITHUB_USERNAME", "abhishektotre"),
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", "")),
        "REPO_VISIBILITY": os.environ.get("REPO_VISIBILITY", "public"), # public or private
        "REPO_PREFIX": os.environ.get("REPO_PREFIX", "") # e.g. "ai-" or ""
    }
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if val:
                        config[key] = val
    return config

CONFIG = load_env()
