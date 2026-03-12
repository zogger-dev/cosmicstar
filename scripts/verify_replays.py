import sc2reader
import concurrent.futures
import os
import time
import glob

# Path to replays
REPLAY_ROOT = r"C:\Users\prade\sc2_data\replays"
WORKERS = 32 # Utilize verify high core count

def verify_replay(path):
    try:
        # load_level=2 loads headers and players, enough to check for file corruption
        # load_level=4 loads all events (much slower)
        sc2reader.load_replay(path, load_level=2)
        return (True, None)
    except Exception as e:
        return (False, str(e))

def main():
    print(f"Scanning for replays in {REPLAY_ROOT}...")
    # Recursive search for .SC2Replay files
    # Using glob might be faster/simpler than os.walk for simple extension matching
    replay_files = glob.glob(os.path.join(REPLAY_ROOT, "**", "*.SC2Replay"), recursive=True)
    
    total_files = len(replay_files)
    print(f"Found {total_files} replays. Starting verification with {WORKERS} workers...")
    
    start_time = time.time()
    
    failed_files = []
    processed_count = 0
    
    # ProcessPoolExecutor is better for CPU-bound tasks like parsing
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # Submit all tasks
        # We process in chunks to avoid creating 64k future objects immediately if possible, 
        # but for simplicity submitting maps is okay for 64k.
        
        # Mapping inputs to results
        results = executor.map(verify_replay, replay_files)
        
        # Iterate results as they come in? map returns iterator in order. 
        # For progress bar we might want as_completed but map is simpler for ordered list handling if needed.
        # Let's use zip to match file with result
        
        for i, (success, error) in enumerate(results):
            processed_count += 1
            if not success:
                print(f"\n[FAIL] {replay_files[i]}: {error}")
                failed_files.append((replay_files[i], error))
            
            # Progress update every 1000 files
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                print(f"Processed {processed_count}/{total_files} ({rate:.1f} files/sec) - Failures: {len(failed_files)}", end='\r')

    total_time = time.time() - start_time
    print(f"\nVerification complete in {total_time:.2f} seconds.")
    print(f"Total Scanned: {total_files}")
    print(f"Total Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for path, err in failed_files[:10]: # Check first 10
            print(f"{path} -> {err}")
        if len(failed_files) > 10:
            print(f"... and {len(failed_files) - 10} more.")
            
        # Write failures to log
        with open("failed_replays.log", "w") as f:
            for path, err in failed_files:
                f.write(f"{path},{err}\n")
        print("Failures written to failed_replays.log")
    else:
        print("All replays are valid!")

if __name__ == "__main__":
    # Windows support for multiprocessing
    # sc2reader uses mpyq which might use some struct unpacking
    main()
