import requests
import zipfile
import os
import sys

# Default to the 3.16.1 pack which is publicly available without API keys
# Newer packs (4.x) typically require Blizzard API credentials.
DEFAULT_URL = "http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_1.zip"
FILENAME = "Replays_3.16.1.zip"
# Use a cache directory outside the workspace to avoid slowing down the IDE
CACHE_DIR = os.path.join(os.path.expanduser("~"), "sc2_data")
EXTRACT_DIR = os.path.join(CACHE_DIR, "replays")
PASSWORD = b"iagreetotheeula"

def download_file(url, filename):
    print(f"Downloading from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length', 0))
        dl = 0
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    dl += len(chunk)
                    f.write(chunk)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl//1024}KB")
    print("\nDownload complete.")

def extract_zip(filename, extract_to, password):
    print(f"Extracting to {extract_to}...")
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    try:
        with zipfile.ZipFile(filename, 'r') as zf:
            zf.extractall(path=extract_to, pwd=password)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    zip_path = os.path.join(CACHE_DIR, FILENAME)
    
    if not os.path.exists(zip_path):
        # check if it is in current dir and move it
        if os.path.exists(FILENAME):
            print(f"Moving {FILENAME} to {CACHE_DIR}...")
            os.rename(FILENAME, zip_path)
        else:
            download_file(DEFAULT_URL, zip_path)
    else:
        print(f"{zip_path} already exists. Skipping download.")
        
    extract_zip(zip_path, EXTRACT_DIR, PASSWORD)
