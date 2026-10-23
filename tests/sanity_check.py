"""Forward/backward sanity check for the policy stubs in cosmicstar.models.

Runs one optimizer step of the Agent (scalar torso + value head) on a batch from
CosmicDataset, or on random data when no processed replays are available.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from cosmicstar.core.dataset import CosmicDataset
from cosmicstar.models.agent import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("COSMICSTAR_PROCESSED", ""))
    args = parser.parse_args()

    print("=== Model Sanity Check ===")

    # 1. Load data
    if args.data_dir and os.path.isdir(args.data_dir):
        dataset = CosmicDataset(args.data_dir)
        if len(dataset) == 0:
            print("Dataset empty. Exiting.")
            return
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=True)))
    else:
        print("No processed data directory given; using dummy data...")
        batch = {"scalars": torch.randn(2, 4)}
    scalars = batch["scalars"].float()
    print(f"Input Shape: {tuple(scalars.shape)}")

    # 2. Initialize model and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Agent().to(device)
    obs = {"scalars": scalars.to(device)}
    with torch.no_grad():
        model(obs)  # materialize lazy layers
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Model initialized on {device}.")

    # 3. One forward/backward pass against a dummy target of zeros
    targets = torch.zeros(scalars.shape[0], 1, device=device)
    before = [p.detach().clone() for p in model.parameters()]
    preds = model(obs)
    loss = torch.mean((preds - targets) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    print(f"Success! Loss: {loss.item():.4f}")
    print(f"Predictions: {preds.detach().reshape(-1).cpu().tolist()}")
    print(f"Parameters updated: {changed}")
    print("Sanity Check Complete.")


if __name__ == "__main__":
    main()
