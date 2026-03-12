import sys
import os
import torch
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
# Add root to path (for dataset.py)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.agent import Agent
from dataset import CosmicDataset

def create_train_state(rng, model, sample_obs):
    """Creates initial `TrainState`."""
    params = model.init(rng, sample_obs)
    tx = optax.adam(1e-3)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_obs, batch_target_val):
    """Train for a single step."""
    def loss_fn(params):
        val_pred = state.apply_fn(params, batch_obs)
        # Dummy value target: 0.0 (just for sanity check)
        loss = jnp.mean((val_pred - batch_target_val) ** 2)
        return loss, val_pred
    
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, predictions

def main():
    print("=== JAX Model Sanity Check ===")
    
    # 1. Load Data
    data_dir = r"C:\Users\prade\sc2_data\processed"
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        # Create dummy data for test if real data missing
        print("Using dummy data...")
        dummy_scalars = torch.randn(2, 4)
        dataloader = [{"scalars": dummy_scalars}]
    else:
        dataset = CosmicDataset(data_dir)
        if len(dataset) == 0:
            print("Dataset empty. Exiting.")
            return
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get one batch
    pytorch_batch = next(iter(dataloader))
    print("Data Batch Loaded.")
    
    # Convert to JAX
    jax_batch = {
        'scalars': jnp.array(pytorch_batch['scalars'].numpy())
    }
    print(f"Input Shape: {jax_batch['scalars'].shape}")
    
    # 2. Initialize Model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    model = Agent()
    print("Model initialized.")
    
    # Initialize State
    state = create_train_state(init_rng, model, jax_batch)
    print("TrainState created.")
    
    # 3. Run Forward/Backward Pass
    # Dummy target values
    batch_size = jax_batch['scalars'].shape[0]
    dummy_targets = jnp.zeros((batch_size, 1))
    
    print("Running training step...")
    state, loss, preds = train_step(state, jax_batch, dummy_targets)
    
    print(f"Success! Loss: {loss:.4f}")
    print(f"Predictions: {preds.reshape(-1)}")
    
    # Check if params changed (simple check)
    # We can't easily check param change without a copy, but if it ran without error, that's 90% of the battle.
    print("Sanity Check Complete.")

if __name__ == "__main__":
    main()
