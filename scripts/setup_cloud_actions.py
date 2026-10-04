"""
Sets up the central GitHub repository and configures GitHub Actions secrets.
"""

import sys
import base64
import requests
from nacl import encoding, public
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from automation.config import CONFIG

token = CONFIG.get("GITHUB_TOKEN")
username = CONFIG.get("GITHUB_USERNAME")
repo_name = "daily-ai-ml-portfolio"

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

def create_or_get_repo():
    print(f"Checking repository '{repo_name}' for user '{username}'...")
    url = f"https://api.github.com/repos/{username}/{repo_name}"
    res = requests.get(url, headers=headers)
    
    if res.status_code == 200:
        print(f"Repository '{repo_name}' already exists.")
        return res.json()
    
    print(f"Creating repository '{repo_name}'...")
    create_url = "https://api.github.com/user/repos"
    payload = {
        "name": repo_name,
        "description": "Autonomous Daily AI, ML, Data Science & Analytics Portfolio Engine. Automatically designs, benchmarks, and deploys standalone production projects every 24 hours.",
        "private": False,
        "auto_init": False
    }
    create_res = requests.post(create_url, json=payload, headers=headers)
    if create_res.status_code == 201:
        print(f"Successfully created repository: {create_res.json().get('html_url')}")
        return create_res.json()
    else:
        raise RuntimeError(f"Failed to create repo: {create_res.status_code} - {create_res.text}")

def set_action_secret(secret_name: str, secret_value: str):
    print(f"Configuring GitHub Actions secret '{secret_name}'...")
    # 1. Get repo public key
    pk_url = f"https://api.github.com/repos/{username}/{repo_name}/actions/secrets/public-key"
    pk_res = requests.get(pk_url, headers=headers)
    if pk_res.status_code != 200:
        raise RuntimeError(f"Failed to get repo public key: {pk_res.status_code} - {pk_res.text}")
    
    pk_data = pk_res.json()
    key_id = pk_data["key_id"]
    public_key_b64 = pk_data["key"]
    
    # 2. Encrypt secret with public key using libsodium SealedBox
    public_key = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder)
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    encrypted_b64 = base64.b64encode(encrypted).decode("utf-8")
    
    # 3. Put secret
    put_url = f"https://api.github.com/repos/{username}/{repo_name}/actions/secrets/{secret_name}"
    put_payload = {
        "encrypted_value": encrypted_b64,
        "key_id": key_id
    }
    put_res = requests.put(put_url, json=put_payload, headers=headers)
    if put_res.status_code in [201, 204]:
        print(f"Successfully configured secret '{secret_name}' in '{repo_name}'!")
    else:
        raise RuntimeError(f"Failed to set secret '{secret_name}': {put_res.status_code} - {put_res.text}")

def main():
    repo = create_or_get_repo()
    set_action_secret("AUTO_GH_TOKEN", token)
    print("\nCloud repository & secret setup complete!")
    print(f"Remote URL: {repo.get('clone_url')}")

if __name__ == "__main__":
    main()
