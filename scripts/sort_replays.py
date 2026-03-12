import os
import mpyq
from s2protocol import versions
from collections import defaultdict

def sort_replays(directory):
    builds = defaultdict(list)
    replays = []
    
    # Recursively find all replays
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.SC2Replay'):
                replays.append(os.path.join(root, f))
    
    print(f"Found {len(replays)} replays. Analyzing headers...")
    
    protocol = versions.latest()
    
    for i, path in enumerate(replays):
        try:
            archive = mpyq.MPQArchive(path)
            contents = archive.header['user_data_header']['content']
            header = protocol.decode_replay_header(contents)
            base_build = header['m_version']['m_baseBuild']
            builds[base_build].append(path)
        except Exception as e:
            print(f"Error parsing header for {path}: {e}")
            
    print("\n--- Summary ---")
    for build, files in sorted(builds.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"Build {build}: {len(files)} replays")
        # Print the first few files to give an example
        for f in files[:3]:
            print(f"  - {os.path.basename(f)}")
            
if __name__ == "__main__":
    import sys
    sort_replays(sys.argv[1])
