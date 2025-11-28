import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path
import os

# Must be identical to the values in train_model.py
MODEL_PATH = 'best_gesture_model.pth'
DATA_DIR = 'data/processed'

WINDOW_SIZE = 8
FEATURES_PER_FRAME = 138
INPUT_DIM = FEATURES_PER_FRAME * WINDOW_SIZE
EMBEDDING_DIM = 128
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25

NUM_SAMPLES_TO_VISUALIZE = 5

# TODO: Move this class to its own file. Its the same as train_model
class GestureDataset(Dataset):
    """A simplified, read-only version of the dataset for testing."""
    def __init__(self, data_dir, window_size, stride):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.windows = []
        self.file_paths = []

        all_files = sorted(list(self.data_dir.rglob('*.npy')))
        for file_path in all_files:
            try:
                # Use mmap_mode for fast length checking
                data = np.load(file_path, mmap_mode='r')
                num_frames = data.shape[0]

                if num_frames >= self.window_size:
                    current_file_idx = len(self.file_paths)
                    self.file_paths.append(file_path)
                    for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
                        self.windows.append((current_file_idx, start_idx))
            except Exception:
                continue # Skip corrupt files

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start_index = self.windows[idx]
        file_path = self.file_paths[file_idx]
        full_data = np.load(file_path)
        window = full_data[start_index : start_index + self.window_size]
        return torch.from_numpy(window.flatten()).float()

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__();self._embedding_dim=embedding_dim;self._num_embeddings=num_embeddings;self._commitment_cost=commitment_cost;self._embedding=nn.Embedding(self._num_embeddings,self._embedding_dim);self._embedding.weight.data.uniform_(-1/self._num_embeddings,1/self._num_embeddings);self.register_buffer('_code_usage',torch.zeros(self._num_embeddings))
    def forward(self, inputs):
        input_shape=inputs.shape;flat_input=inputs.view(-1,self._embedding_dim);distances=(torch.sum(flat_input**2,dim=1,keepdim=True)+torch.sum(self._embedding.weight**2,dim=1)-2*torch.matmul(flat_input,self._embedding.weight.t()));encoding_indices=torch.argmin(distances,dim=1).unsqueeze(1);encodings=torch.zeros(encoding_indices.shape[0],self._num_embeddings,device=inputs.device).scatter_(1,encoding_indices,1);quantized=torch.matmul(encodings,self._embedding.weight).view(input_shape);loss=F.mse_loss(quantized,inputs.detach())+self._commitment_cost*F.mse_loss(quantized.detach(),inputs);quantized=inputs+(quantized-inputs).detach();perplexity=torch.exp(-torch.sum(torch.mean(encodings,dim=0)*torch.log(torch.mean(encodings,dim=0)+1e-10)));return loss,quantized,perplexity,encoding_indices.view(input_shape[0],-1)

class VQVAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__();self._encoder=nn.Sequential(nn.Linear(input_dim,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,embedding_dim));self._vq=VectorQuantizer(num_embeddings,embedding_dim,commitment_cost);self._decoder=nn.Sequential(nn.Linear(embedding_dim,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,input_dim))
    def forward(self,x):
        z=self._encoder(x);vq_loss,quantized,perplexity,_=self._vq(z);x_recon=self._decoder(quantized);return F.mse_loss(x_recon,x),vq_loss,perplexity
    def get_tokens_and_reconstructions(self,x):
        z=self._encoder(x);_,q,_,i=self._vq(z);x_r=self._decoder(q);return x_r,F.mse_loss(x_r,x),i

# Draw a link between each landmark
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

def plot_hand(ax, landmarks_3d, color='blue'):
    """Plots a single hand skeleton if it's visible."""
    # Check if the hand data is not just zeros
    if np.any(landmarks_3d):
        ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2], c=color, marker='o', s=10)
        for start_idx, end_idx in HAND_CONNECTIONS:
            ax.plot([landmarks_3d[start_idx,0],landmarks_3d[end_idx,0]],
                    [landmarks_3d[start_idx,1],landmarks_3d[end_idx,1]],
                    [landmarks_3d[start_idx,2],landmarks_3d[end_idx,2]], color)

def calculate_global_positions(window_data):
    """
    Re-integrates wrist velocity for both hands from a 138-feature window.
    """
    frames = window_data.reshape(WINDOW_SIZE, FEATURES_PER_FRAME)
    
    # Extract pose and velocity for each hand
    left_poses_norm = frames[:, 0:63].reshape(WINDOW_SIZE, 21, 3)
    left_vels = frames[:, 63:66]
    right_poses_norm = frames[:, 69:132].reshape(WINDOW_SIZE, 21, 3)
    right_vels = frames[:, 132:135]
    
    # Integrate velocities to get global wrist positions
    left_wrist_pos = np.cumsum(left_vels, axis=0)
    right_wrist_pos = np.cumsum(right_vels, axis=0)
    
    # Add global positions back to normalized poses
    left_poses_global = left_poses_norm + left_wrist_pos[:, np.newaxis, :]
    right_poses_global = right_poses_norm + right_wrist_pos[:, np.newaxis, :]
    
    return left_poses_global, right_poses_global

def animate_with_global_motion(original_window, reconstructed_window, token_id, sample_index):
    """Creates a side-by-side animation for one or two hands."""
    orig_left, orig_right = calculate_global_positions(original_window)
    recon_left, recon_right = calculate_global_positions(reconstructed_window)

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Determine plot limits from all original motion
    all_orig_coords = np.concatenate([orig_left, orig_right])
    min_coords = all_orig_coords.min(axis=(0, 1)); max_coords = all_orig_coords.max(axis=(0, 1))
    
    def update(frame):
        ax1.cla(); ax2.cla()

        plot_hand(ax1, orig_left[frame], color='blue')
        plot_hand(ax1, orig_right[frame], color='cyan')
        
        plot_hand(ax2, recon_left[frame], color='red')
        plot_hand(ax2, recon_right[frame], color='magenta')

        ax1.set_title(f"Original Motion (Frame {frame+1}/{WINDOW_SIZE})")
        ax2.set_title(f"Reconstructed Motion (Token ID: {token_id})")
        for ax in [ax1, ax2]:
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_xlim([min_coords[0], max_coords[0]]); ax.set_ylim([min_coords[1], max_coords[1]]); ax.set_zlim([min_coords[2], max_coords[2]])
            ax.view_init(elev=20, azim=120)

    fig.suptitle(f'Sample #{sample_index}', fontsize=16)
    anim = FuncAnimation(fig, update, frames=WINDOW_SIZE, interval=120, repeat=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


def test_model():
    """Main function to load model and run visualizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {MODEL_PATH}")
    model = VQVAE(INPUT_DIM, EMBEDDING_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'. Please train the model first.")
        return
    model.eval()

    print("Loading dataset...")
    test_dataset = GestureDataset(data_dir=DATA_DIR, window_size=WINDOW_SIZE, stride=WINDOW_SIZE * 2) # Use a large stride to get diverse samples
    if len(test_dataset) == 0:
        print(f"ERROR: No data found in '{DATA_DIR}'.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=NUM_SAMPLES_TO_VISUALIZE, shuffle=True)

    with torch.no_grad():
        original_data = next(iter(test_loader)).to(device)
        reconstructed_data, recon_error, token_indices = model.get_tokens_and_reconstructions(original_data)
        
        print(f"\nTest Batch Reconstruction Error: {recon_error.item():.5f}")
        print("-" * 40)

        original_data = original_data.cpu().numpy()
        reconstructed_data = reconstructed_data.cpu().numpy()
        token_indices = token_indices.cpu().numpy()

        print(f"Displaying {NUM_SAMPLES_TO_VISUALIZE} animated comparisons...")
        for i in range(len(original_data)):
            original_window = original_data[i]
            reconstructed_window = reconstructed_data[i]
            token_id = token_indices[i][0]
            print(f"-> Showing Sample #{i}: Mapped to Token ID -> {token_id}")
            animate_with_global_motion(original_window, reconstructed_window, token_id, i)

if __name__ == '__main__':
    test_model()