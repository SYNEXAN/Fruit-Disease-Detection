"""
Utility script to push Fruit_Detect project to a Hugging Face Space
Usage:
    python push_to_hf.py [space_id]
Example:
    python push_to_hf.py thietnt/Fruit_Disease_Detection
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
user = api.whoami()["name"]

repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{user}/Fruit_Disease_Detection"

print(f"Target Hugging Face Space: {repo_id}")
folder_path = Path(__file__).resolve().parent

ignore_patterns = [
    "__pycache__/**",
    "*.pyc",
    "*.zip",
    "*.tmp",
    ".git/**",
    ".venv/**",
    "push_to_hf.py"
]

try:
    print(f"Uploading files from {folder_path} to Space {repo_id}...")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        commit_message="Deploy Fruit Disease Detection with EfficientNetB0"
    )
    print("\n✅ Successfully deployed to Hugging Face Spaces!")
    print(f"👉 Live App URL: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print("\n❌ Error uploading to Space:", e)
    print("Tip: If the Space does not exist yet, please create it at https://huggingface.co/new-space (SDK: Gradio), then run this script again.")
