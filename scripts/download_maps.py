import argparse
import os
import zipfile
import requests
import concurrent.futures
from tqdm import tqdm

# Imports for dependency scanning
try:
    import sc2reader
except ImportError:
    sc2reader = None

# Configuration
BASE_URL = "http://blzdistsc2-a.akamaihd.net/MapPacks"
MELEE_PACK = "Melee.zip"
PASSWORD = b"iagreetotheeula"

# Depot Configuration
# This endpoint serves individual assets by hash
DEPOT_URL_TEMPLATE = "http://us.depot.battle.net:1119/{hash}.{ext}"

# Standard paths
DEFAULT_SC2_DIR = r"C:\Program Files (x86)\StarCraft II"
DEFAULT_MAPS_DIR = os.path.join(DEFAULT_SC2_DIR, "Maps")
USER_MAPS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "StarCraft II", "Maps")

def _copy_stream_with_progress(src, dst, total_size):
    """Helper to copy stream with progress bar."""
    block_sz = 8192
    downloaded = 0
    
    with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, leave=False) as bar:
        while True:
            buffer = src.read(block_sz)
            if not buffer:
                break
            
            downloaded += len(buffer)
            dst.write(buffer)
            bar.update(len(buffer))

def download_file(url, target_path, verbose=True):
    """Download a file using requests."""
    if verbose:
        print(f"Downloading: {url}")
    try:
        response = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            # Re-use helper logic but adapted for requests iterator
             with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, desc=os.path.basename(target_path), leave=verbose) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        return True
    
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
            # Quietly fail on 404 for ladder/depot iteration checks
            pass 
        else:
            print(f"  Download Error: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def _extract_chunk(zip_path, file_list, extract_to, pwd):
    """Helper for parallel extraction."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in file_list:
            try:
                zf.extract(member, path=extract_to, pwd=pwd)
            except Exception as e:
                print(f"Error extracting {member.filename}: {e}")

def extract_zip_parallel(zip_path, extract_to, workers=16):
    """Extracts zip file in parallel."""
    print(f"Extracting {os.path.basename(zip_path)} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_members = zf.infolist()
            
        total_files = len(all_members)
        if total_files == 0: return True

        if total_files < workers: workers = total_files
        chunk_size = (total_files // workers) + 1
        chunks = [all_members[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_extract_chunk, zip_path, chunk, extract_to, PASSWORD) for chunk in chunks]
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Unzipping"): pass
                
        return True
    except Exception as e:
        print(f"  Extraction Failed: {e}")
        return False

def _download_from_depot(resource_hash, ext, target_dir):
    """Downloads a resource from the Battle.net depot by hash."""
    url = DEPOT_URL_TEMPLATE.format(hash=resource_hash, ext=ext)
    target_path = os.path.join(target_dir, f"{resource_hash}.{ext}")
    
    if os.path.exists(target_path):
        return True, target_path

    # print(f"  Fetching dependency {resource_hash} from Depot...")
    if download_file(url, target_path, verbose=False):
        return True, target_path
    return False, None

def scan_and_download_deps(replay_dir, maps_dir):
    """Scans replays for missing maps and downloads them from Battle.net."""
    if not sc2reader:
        print("ERROR: sc2reader not installed. Cannot scan replays.")
        return

    print(f"\n[Scanning Replays in {replay_dir}]")
    
    # Store downloaded maps in a dedicated folder or directly in maps_dir/Downloaded
    download_dir = os.path.join(maps_dir, "Downloaded")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Find replays
    replays = []
    for root, _, files in os.walk(replay_dir):
        for f in files:
            if f.endswith(".SC2Replay"):
                replays.append(os.path.join(root, f))
                
    print(f"  Found {len(replays)} replays. Checking map dependencies...")
    
    count = 0
    # Use tqdm for scan progress
    for path in tqdm(replays, desc="Verifying Maps"):
        try:
            # We only need the header to get map info
            replay = sc2reader.load_replay(path, load_level=1)
            
            map_hash = replay.map_hash
            map_name = replay.map_name
            
            # 1. Sanitation: Create a filename-safe map name
            safe_name = "".join([c for c in map_name if c.isalnum() or c in (' ', '_', '-')]).strip()
            if not safe_name: safe_name = f"Map_{map_hash}"
            
            target_map_path = os.path.join(download_dir, f"{safe_name}.SC2Map")
            
            # 2. Check if already exists
            if os.path.exists(target_map_path):
                continue
                
            # 3. Check if exists in Ladder/Melee folders (approximate check by name?)
            # This is expensive to scan fully. We assume if it's not in Downloaded, and user
            # hasn't downloaded usage packs, we might redownload.
            # But downloading from depot is safe (idempotent).
            
            tqdm.write(f"  Missing Map for {os.path.basename(path)}: {map_name} ({map_hash})")
            
            # 4. Download (.s2ma -> .SC2Map)
            success, tmp_path = _download_from_depot(map_hash, "s2ma", download_dir)
            
            if success and tmp_path:
                try:
                    # Rename hash-based file to human readable .SC2Map
                    if not os.path.exists(target_map_path):
                        os.rename(tmp_path, target_map_path)
                        tqdm.write(f"    -> Downloaded: {safe_name}.SC2Map")
                        count += 1
                    else:
                         # Race condition or previous run left it
                         os.remove(tmp_path) 
                except OSError as e:
                    tqdm.write(f"    Error renaming: {e}")
            else:
                 tqdm.write(f"    -> FAILED to download map from depot.")

        except Exception as e:
            # tqdm.write(f"  Error parsing {os.path.basename(path)}: {e}")
            pass
            
    print(f"\nDependency Scan Complete. Downloaded {count} missing maps to {download_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Download StarCraft II Maps (Ladder, Melee, and Dependencies).")
    parser.add_argument("--maps-dir", default=None, help="Target directory for maps.")
    parser.add_argument("--exclude-ladder", action="store_true", help="Skip downloading ladder maps.")
    parser.add_argument("--exclude-melee", action="store_true", help="Skip downloading the standard Melee pack.")
    parser.add_argument("--years", nargs="+", type=int, default=range(2017, 2026), help="Years to download ladder maps for.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep downloaded zip files.")
    parser.add_argument("--workers", type=int, default=16, help="Number of threads for extraction.")
    parser.add_argument("--scan-replays", default=None, help="Directory of replays to scan for missing maps.")
    
    args = parser.parse_args()
    
    # 1. Determine Directory
    target_dir = args.maps_dir
    if not target_dir:
        target_dir = DEFAULT_MAPS_DIR
        if not os.path.exists(DEFAULT_SC2_DIR):
             target_dir = USER_MAPS_DIR
    
    if not os.path.exists(target_dir):
        try: os.makedirs(target_dir)
        except: 
            target_dir = USER_MAPS_DIR
            os.makedirs(target_dir, exist_ok=True)

    print(f"Target Maps Directory: {target_dir}")
    
    # 2. Dependency Scan
    if args.scan_replays:
        scan_and_download_deps(args.scan_replays, target_dir)
    
    # 3. Standard Packs
    # Only run standard downloads if not EXPLICITLY scanning (unless user didn't exclude them)
    # Actually, running scan doesn't prevent wanting ladder maps.
    
    if not args.exclude_melee:
        print("\n[Processing Standard Melee Pack]")
        url = f"{BASE_URL}/{MELEE_PACK}"
        zip_path = os.path.join(target_dir, MELEE_PACK)
        
        if download_file(url, zip_path):
            extract_zip_parallel(zip_path, target_dir, workers=args.workers)
            if not args.keep_zips:
                os.remove(zip_path)
    
    if not args.exclude_ladder:
        print("\n[Processing Ladder Maps]")
        count = 0
        for year in args.years:
            for season in range(1, 6):
                filename = f"Ladder{year}Season{season}.zip"
                url = f"{BASE_URL}/{filename}"
                zip_path = os.path.join(target_dir, filename)
                
                if os.path.exists(zip_path):
                    print(f"Skipping {filename} (Zip exists)")
                    continue
                
                if download_file(url, zip_path):
                    if extract_zip_parallel(zip_path, target_dir, workers=args.workers):
                        count += 1
                        if not args.keep_zips:
                            try: os.remove(zip_path)
                            except: pass
        print(f"\nLadder processing complete. Downloaded {count} new packs.")

if __name__ == "__main__":
    main()
