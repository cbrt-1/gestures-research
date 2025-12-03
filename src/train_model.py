import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
import sys

# Import your model (Assumes model/vqvae.py exists)
from model.vqvae import VQVAE

# --- Configuration ---
DATA_DIR = 'data/processed'
MODEL_SAVE_PATH = 'best_gesture_model.pth'
CHECKPOINT_PATH = 'checkpoint.pth'

# Hardware & Training Hyperparameters
BATCH_SIZE = 2048  # Increased batch size (efficient for RTX 4070 with AMP)
LEARNING_RATE = 1e-4
NUM_UPDATES = 200000

# VQ-VAE Specifics
WINDOW_SIZE = 8
STRIDE = 4
COMMITMENT_COST = 0.25
RESET_CODEBOOK_EVERY = 1000
VALIDATE_EVERY = 1000
CHECKPOINT_EVERY = 1000

# Feature Dimensions
FEATURES_PER_FRAME = 138
INPUT_DIM = FEATURES_PER_FRAME * WINDOW_SIZE
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 512

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict

class FastGestureDataset(Dataset):
    """
    Loads all data into RAM using the existing pickle index.
    Optimized to load files once instead of per-window.
    """
    def __init__(self, data_dir, window_size, stride):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        
        # RAM cache file
        self.cache_path = self.data_dir / f"cached_dataset_w{window_size}_s{stride}.pt"
        
        # Original pickle index
        pickle_path = self.data_dir / f"dataset_index_w{window_size}_s{stride}.pkl"
        
        if self.cache_path.exists():
            print(f"Loading consolidated dataset from {self.cache_path}...")
            self.data = torch.load(self.cache_path)
            print(f"✓ Loaded {len(self.data)} windows from RAM.")
        elif pickle_path.exists():
            print(f"Found existing index: {pickle_path}")
            self._build_cache_from_pickle_fast(pickle_path)
        else:
            raise RuntimeError(f"No data found! Need either:\n"
                             f"  - {self.cache_path}\n"
                             f"  - {pickle_path}")

    def _build_cache_from_pickle_fast(self, pickle_path):
        """
        Optimized version: Load each file once, extract all its windows.
        """
        print("Building RAM cache (optimized file loading)...")
        
        # Load the pickle index
        with open(pickle_path, 'rb') as f:
            cache = pickle.load(f)
            windows = cache['windows']
            file_paths = cache['file_paths']
        
        print(f"Found {len(windows)} windows from {len(file_paths)} files")
        
        # Group windows by file for batch processing
        print("Grouping windows by file...")
        file_to_windows = defaultdict(list)
        for window_idx, (file_idx, start_idx) in enumerate(windows):
            file_to_windows[file_idx].append((window_idx, start_idx))
        
        # Pre-allocate tensor (faster than appending)
        print(f"Pre-allocating tensor for {len(windows)} windows...")
        feature_dim = self.window_size * 138  # FEATURES_PER_FRAME
        self.data = torch.zeros(len(windows), feature_dim, dtype=torch.float32)
        
        # Load each file once and extract all windows
        print("Loading files and extracting windows...")
        for file_idx in tqdm(sorted(file_to_windows.keys()), desc="Processing files"):
            try:
                file_path = file_paths[file_idx]
                
                # Load full file ONCE
                full_data = np.load(file_path)
                
                # Extract all windows from this file
                for window_idx, start_idx in file_to_windows[file_idx]:
                    window = full_data[start_idx : start_idx + self.window_size]
                    self.data[window_idx] = torch.from_numpy(window.flatten())
                    
            except Exception as e:
                print(f"Error loading file {file_idx}: {e}")
        
        print(f"Saving cache to {self.cache_path}...")
        torch.save(self.data, self.cache_path)
        print(f"✓ Cache built: {len(self.data)} windows")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    # --- 1. Hardware Setup ---
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for RTX 40-series
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # --- 2. Data Loading (Fast RAM Version) ---
    print("\n=== Initializing Dataset ===")
    dataset = FastGestureDataset(DATA_DIR, WINDOW_SIZE, STRIDE)
    
    val_len = len(dataset) // 10
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Since data is in RAM, num_workers=0 is usually fastest (no IPC overhead)
    # If using 'persistent_workers', set to 2 or 4.
    num_workers = 0 
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True # Fast transfer to GPU
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE * 2,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # --- 3. Model & Optimizer ---
    print("\n=== Initializing Model ===")
    model = VQVAE(INPUT_DIM, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    
    # Optional: Torch Compile (Try disabling if startup fails)
    if hasattr(torch, 'compile') and os.name != 'nt': 
        # Windows (nt) compile support is sometimes experimental, skipping for stability
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=False)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_UPDATES, eta_min=LEARNING_RATE/10)
    
    # MIXED PRECISION SCALER (Speed Boost)
    scaler = torch.cuda.amp.GradScaler()

    # --- 4. Resume Logic ---
    start_step = 1
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n=== Resuming from {CHECKPOINT_PATH} ===")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        # Handle compiled vs non-compiled state keys
        state_dict = ckpt['model_state_dict']
        if hasattr(model, '_orig_mod'):
            # If current model is compiled but checkpoint wasn't (or vice versa), handle prefix
            pass # Usually torch handles this, or we load into model._orig_mod
            
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            
        start_step = ckpt.get('step', 1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_step}")

    # --- 5. Training Loop ---
    print(f"\n=== Starting Training (Steps {start_step} to {NUM_UPDATES}) ===")
    model.train()
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(start_step, NUM_UPDATES + 1), desc="Training", initial=start_step-1, total=NUM_UPDATES)
    
    for i in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Non-blocking transfer to GPU
        batch = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # --- Mixed Precision Forward Pass ---
        with torch.cuda.amp.autocast():
            recon_loss, vq_loss, perplexity = model(batch)
            total_loss = recon_loss + vq_loss
        
        # --- Mixed Precision Backward Pass ---
        scaler.scale(total_loss).backward()
        
        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Logging
        if i % 100 == 0:
            pbar.set_postfix({
                "Recon": f"{recon_loss.item():.4f}",
                "VQ": f"{vq_loss.item():.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
        # Dead Code Revival
        if i > 0 and i % RESET_CODEBOOK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    encoded_batch = model.encode(batch) 
                    model.reset_dead_codes(encoded_batch)
            model.train()

        # Validation
        if i % VALIDATE_EVERY == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        r_loss, v_loss, _ = model(val_batch)
                        val_losses.append((r_loss + v_loss).item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            tqdm.write(f"\n[Step {i}] Val Loss: {avg_val_loss:.5f}")

            # Save Best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save(save_model.state_dict(), MODEL_SAVE_PATH)
                tqdm.write(f"✓ Best model saved")
            
            model.train()

        # Checkpoint
        if i % CHECKPOINT_EVERY == 0:
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'step': i,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss
            }, CHECKPOINT_PATH)

    print("\n=== Training Complete ===")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()