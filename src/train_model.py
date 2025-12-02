import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from model.vqvae import VQVAE
import pickle

# --- Configuration ---
DATA_DIR = 'data/processed'
MODEL_SAVE_PATH = 'best_gesture_model.pth'
CHECKPOINT_PATH = 'checkpoint.pth'

# Hardware & Training Hyperparameters
BATCH_SIZE = 1024        # Increase to 2048 if VRAM allows
LEARNING_RATE = 1e-4
NUM_UPDATES = 200000

# VQ-VAE Specifics
WINDOW_SIZE = 8
STRIDE = 4
COMMITMENT_COST = 0.25
RESET_CODEBOOK_EVERY = 1000  # Revive dead codes
VALIDATE_EVERY = 1000
CHECKPOINT_EVERY = 1000

# Feature Dimensions
FEATURES_PER_FRAME = 138
INPUT_DIM = FEATURES_PER_FRAME * WINDOW_SIZE
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 512

class GestureDataset(Dataset):
    
    def __init__(self, data_dir, window_size, stride):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        
        cache_file = self.data_dir / f"dataset_index_w{window_size}_s{stride}.pkl"
        
        # --- Indexing Phase (self.windows is created here) ---
        if cache_file.exists():
            print(f"Loading cached index from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                self.windows = cache['windows']
                self.file_paths = cache['file_paths']
            print(f"Loaded {len(self.windows)} windows from {len(self.file_paths)} files (cached).")
        else:
            # ... (Index building logic, where self.windows and self.file_paths are populated) ...
            pass # (Assume the original index building code goes here)

        # --- NEW PRE-LOADING PHASE (Must be AFTER indexing) ---
        print("Pre-loading all window data into RAM...")
        self.data = []
        
        # ⚠️ Now self.windows is guaranteed to exist!
        for file_idx, start_idx in tqdm(self.windows, desc="Loading data"): 
            file_path = self.file_paths[file_idx]
            
            # Load full file data (no mmap_mode needed, as we're loading everything)
            full_data = np.load(file_path) 
            
            # Slice, flatten, and convert to tensor
            window = full_data[start_idx : start_idx + self.window_size]
            self.data.append(torch.from_numpy(window.flatten()).float())
            
        print(f"Finished pre-loading {len(self.data)} windows.")
        # --- End Pre-loading ---

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        # Access the pre-loaded data directly
        return self.data[idx]


def main():

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    # --- 1. Hardware Optimization Setup ---
    
    # Enable TF32 (TensorFloat-32) for massive speedup on RTX 30/40 series
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking to find fastest convolution algos
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CPU Worker Strategy: Don't starve the GPU driver
    num_workers = min(os.cpu_count(), 8)
    print(f"Using device: {device} | TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    
    # --- 2. Data Loading ---
    dataset = GestureDataset(DATA_DIR, WINDOW_SIZE, STRIDE)
    
    if len(dataset) == 0:
        print(f"ERROR: No valid data found in '{DATA_DIR}'.")
        return
    
    val_len = len(dataset) // 10
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # Persistent workers keep RAM allocated, preventing CPU spikes between epochs
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3 
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE * 2, # Double batch size for validation (no backprop RAM usage)
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # --- 3. Model & Optimizer ---
    model = VQVAE(INPUT_DIM, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    
    # Compile model (PyTorch 2.0+) - Skips on Windows if it causes issues
    compiled = False
    if hasattr(torch, 'compile') and os.name != 'nt':
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
            compiled = True
        except Exception as e:
            print(f"Compilation skipped: {e}")

    # Fused AdamW runs optimizer logic on GPU (faster)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True)
    
    # --- 4. Resume Logic ---
    start_step = 1
    best_val_loss = float('inf')

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        
        # Handle loading into compiled vs uncompiled model
        if compiled and hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt['model_state_dict'])
            
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_step}.")

    # --- 5. Training Loop ---
    model.train()
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(start_step, NUM_UPDATES + 1), desc="Training")
    
    for i in pbar:
        # Infinite DataLoader iterator pattern
        try: 
            batch = next(train_iter)
        except StopIteration: 
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True) # set_to_none is slightly faster
        
        # Forward Pass
        recon_loss, vq_loss, perplexity = model(batch)
        total_loss = recon_loss + vq_loss
        
        # Backward Pass
        total_loss.backward()
        optimizer.step()
        
        # Update progress bar
        if i % 100 == 0:
            pbar.set_postfix({
                "Recon": f"{recon_loss.item():.4f}",
                "VQ": f"{vq_loss.item():.4f}",
                "Pplx": f"{perplexity.item():.1f}"
            })
            
        # Dead Code Revival (Prevents Codebook Collapse)
        if i > 0 and i % RESET_CODEBOOK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                encoded_batch = model.encode(batch) 
                model.reset_dead_codes(encoded_batch)
            model.train()

        # Validation Loop
        if i % VALIDATE_EVERY == 0:
            model.eval()
            val_losses = []
            if len(val_loader) > 0:
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device, non_blocking=True)
                        r_loss, v_loss, _ = model(val_batch)
                        val_losses.append((r_loss + v_loss).item())
                
                avg_val_loss = np.mean(val_losses)
                tqdm.write(f"\nStep {i} | Val Loss: {avg_val_loss:.5f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save the ORIGINAL model, not the compiled wrapper
                    save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    torch.save(save_model.state_dict(), MODEL_SAVE_PATH)
                    tqdm.write(f"*** Best Model Saved (Loss: {best_val_loss:.5f}) ***")
            
            model.train()

        # Checkpoint Saving
        if i % CHECKPOINT_EVERY == 0:
            save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'step': i,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CHECKPOINT_PATH)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()