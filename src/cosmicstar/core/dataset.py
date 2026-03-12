import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os

class CosmicDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (string): Directory with all the .pt files.
        """
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        print(f"Found {len(self.files)} processed replay files.")
        
        # Load all samples into memory? 
        # For 64k replays, this will OOM. We need to lazy load or index.
        # But our .pt files are "Replay Chunks" (one file = one replay).
        # Standard approach: Create an index of (File_Index, Sample_Index).
        
        self.index_map = [] # List of (file_path, inner_index)
        
        for f in self.files:
            try:
                # We have to load the file to know how many samples it has?
                # Optimization: Save metadata separately, or accept slow startup.
                # For now, let's load them (since we only have 1 file for testing).
                data = torch.load(f, weights_only=False)
                for i in range(len(data)):
                    self.index_map.append((f, i))
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        print(f"Total Samples indexed: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, inner_idx = self.index_map[idx]
        
        # Optimization: Cache the last loaded file to avoid repetitive I/O if batching is poor
        # But PyTorch DataLoader usually uses multiple workers.
        # Just standard load for now.
        
        chunk_data = torch.load(file_path, weights_only=False)
        sample = chunk_data[inner_idx]
        
        # Unpack
        scalars = torch.tensor(sample['scalars'])
        unit_count = sample['unit_count'] # This is an int, maybe we need the actual unit list later
        action_ids = torch.tensor(sample['action_ids']).long()
        
        # In the real model, we would pad the unit list here.
        # For verification, just return what we have.
        
        # Basic transformation: Pad actions to fixed length?
        # Let's just return the first action for simplicity if using CrossEntropy
        target_action = -1
        if len(action_ids) > 0:
            target_action = action_ids[0]
        else:
            target_action = 0 # No-op
            
        return dict(
            scalars=scalars,
            target_action=target_action
        )

if __name__ == "__main__":
    # Test the dataset
    ds = CosmicDataset(r"C:\Users\prade\sc2_data\processed")
    if len(ds) > 0:
        print("Sample 0:", ds[0])
        
        dl = DataLoader(ds, batch_size=2, shuffle=True)
        for batch in dl:
            print("Batch:", batch)
            break
