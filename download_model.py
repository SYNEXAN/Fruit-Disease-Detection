"""
Script: download_model.py
Purpose: Automatically verify and download ConvNeXt-V2 CMTL v6 model weights (114.7 MB)
         directly from GitHub LFS Media Storage if the server environment (e.g. Render)
         does not have `git-lfs` installed.
"""

import os
import sys
import time
from pathlib import Path
import urllib.request

MODEL_URL = "https://media.githubusercontent.com/media/SYNEXAN/Fruit-Disease-Detection/main/model/convnextv2_cmtl_v6_best.pth"
MIN_EXPECTED_SIZE = 50 * 1024 * 1024  # At least 50 MB (full weights ~114.7 MB)

def ensure_model_downloaded() -> bool:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "convnextv2_cmtl_v6_best.pth"
    
    if model_path.is_file():
        file_size = model_path.stat().st_size
        if file_size >= MIN_EXPECTED_SIZE:
            print(f"[Model Check] Found valid model weights: {model_path} ({file_size / (1024*1024):.2f} MB). Skipping download.")
            return True
        else:
            print(f"[Model Check] File {model_path} is only {file_size} bytes (Git LFS pointer). Downloading full weights...")
    else:
        print(f"[Model Check] Model file {model_path} not found. Downloading from GitHub Media...")

    temp_path = model_path.with_suffix(".tmp")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FruitDiseaseDetection/1.0"}
    req = urllib.request.Request(MODEL_URL, headers=headers)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(temp_path, "wb") as f:
            total_size = int(resp.getheader("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            print(f"[Model Download] Starting download from: {MODEL_URL}")
            print(f"[Model Download] Total size: {total_size / (1024*1024):.2f} MB")

            last_log = time.time()
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if time.time() - last_log > 3.0 or downloaded == total_size:
                    percent = (downloaded / total_size * 100) if total_size else 0
                    print(f"[Model Download] Downloaded: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)")
                    last_log = time.time()

        if temp_path.stat().st_size >= MIN_EXPECTED_SIZE:
            if model_path.exists():
                model_path.unlink()
            temp_path.rename(model_path)
            duration = time.time() - start_time
            print(f"[Model Download] Completed in {duration:.1f}s: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"[Model Download ERROR] Downloaded size is insufficient: {temp_path.stat().st_size} bytes")
            if temp_path.exists():
                temp_path.unlink()
            return False

    except Exception as e:
        print(f"[Model Download ERROR] Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

if __name__ == "__main__":
    success = ensure_model_downloaded()
    if not success:
        sys.exit(1)
    sys.exit(0)
