import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self.register_buffer('_code_usage', torch.zeros(self._num_embeddings))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        if self.training:
            self._code_usage += torch.sum(encodings, dim=0)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encoding_indices.view(input_shape[0], -1)

    @torch.no_grad()
    def reset_dead_codes(self, encoder_outputs):
        """Resets unused codebook vectors to random encoder outputs."""
        dead_mask = (self._code_usage == 0)
        num_dead = torch.sum(dead_mask).item()
        
        if num_dead > 0:
            flat_outputs = encoder_outputs.view(-1, self._embedding_dim)
            indices = torch.randint(0, len(flat_outputs), (num_dead,))
            self._embedding.weight.data[dead_mask] = flat_outputs[indices].to(self._embedding.weight.dtype)
            print(f"Reset {num_dead} dead codebook vectors")
            self._code_usage.zero_()

class VQVAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeddings, commitment_cost, dropout=0.0):
        super(VQVAE, self).__init__()
        
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        self._vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self._decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self._encoder(x)
        vq_loss, quantized, perplexity, _ = self._vq(z)
        x_recon = self._decoder(quantized)
        recon_error = F.mse_loss(x_recon, x)
        return recon_error, vq_loss, perplexity

    def get_tokens_and_reconstructions(self, x):
        z = self._encoder(x)
        _, quantized, _, indices = self._vq(z)
        x_recon = self._decoder(quantized)
        return x_recon, F.mse_loss(x_recon, x), indices
        
    def encode(self, x):
        return self._encoder(x)
        
    def reset_dead_codes(self, encoder_outputs):
        self._vq.reset_dead_codes(encoder_outputs)