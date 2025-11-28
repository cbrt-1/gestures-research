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

# Config
DATA_DIR = 'data/processed'
MODEL_SAVE_PATH = 'best_gesture_model.pth'
CHECKPOINT_PATH = 'checkpoint.pth'

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
NUM_UPDATES = 200000
WINDOW_SIZE = 8
STRIDE = 4
COMMITMENT_COST = 0.25
RESET_CODEBOOK_EVERY = 1000
VALIDATE_EVERY = 1000
CHECKPOINT_EVERY = 1000

# Feature vector dimensions (138 features per frame)
FEATURES_PER_FRAME = 138
INPUT_DIM = FEATURES_PER_FRAME * WINDOW_SIZE
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 512


class GestureDataset(Dataset):
    """Scans a directory for .npy files and creates a dataset of sliding windows."""
    def __init__(self, data_dir, window_size, stride):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.file_paths = []

        print(f"Indexing data in {self.data_dir}...")
        all_files = sorted(list(self.data_dir.rglob('*.npy')))
        
        for file_path in tqdm(all_files, desc="Indexing files"):
            try:
                # Get frame count without loading the full file into memory
                data = np.load(file_path, mmap_mode='r')
                num_frames = data.shape[0]

                if num_frames >= self.window_size:
                    current_file_idx = len(self.file_paths)
                    self.file_paths.append(file_path)
                    for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
                        self.windows.append((current_file_idx, start_idx))
            except Exception as e:
                print(f"Warning: Skipping corrupt or unreadable file {file_path}. Reason: {e}")

        print(f"Found {len(self.windows)} total windows from {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start_index = self.windows[idx]
        file_path = self.file_paths[file_idx]
        
        full_data = np.load(file_path)
        window = full_data[start_index : start_index + self.window_size]
        
        return torch.from_numpy(window.flatten()).float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_workers = min(os.cpu_count(), 8)
    print(f"Using device: {device} with {num_workers} data loader workers.")
    
    dataset = GestureDataset(DATA_DIR, WINDOW_SIZE, STRIDE)
    
    if len(dataset) == 0:
        print(f"ERROR: No valid .npy files found in '{DATA_DIR}'.")
        return
    
    val_len = len(dataset) // 10
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE * 2,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split: {len(train_ds)} train windows, {len(val_ds)} validation windows.")

    model = VQVAE(INPUT_DIM, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    start_step = 1

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_step}.")

    model.train()
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(start_step, NUM_UPDATES + 1), desc="Training")
    for i in pbar:
        try: 
            batch = next(train_iter)
        except StopIteration: 
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        batch = batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        recon_loss, vq_loss, perplexity = model(batch)
        total_loss = recon_loss + vq_loss
        total_loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            pbar.set_postfix({
                "Recon": f"{recon_loss.item():.5f}",
                "VQ": f"{vq_loss.item():.5f}",
                "Pplx": f"{perplexity.item():.1f}"
            })
            
        if i > 0 and i % RESET_CODEBOOK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                encoded_batch = model.encode(batch) 
                model.reset_dead_codes(encoded_batch)
            model.train()

        if i % VALIDATE_EVERY == 0 and len(val_loader) > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device, non_blocking=True)
                    r_loss, v_loss, _ = model(val_batch)
                    val_losses.append((r_loss + v_loss).item())
            
            avg_val_loss = np.mean(val_losses)
            
            tqdm.write("-" * 50)
            tqdm.write(f"Validation @ Step {i} | Avg Loss: {avg_val_loss:.5f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                tqdm.write(f"*** New best model saved (Val Loss: {best_val_loss:.5f}) ***")
            tqdm.write("-" * 50)
            
            model.train()

        if i % CHECKPOINT_EVERY == 0:
            torch.save({
                'step': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, CHECKPOINT_PATH)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()