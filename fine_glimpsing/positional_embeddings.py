import torch
import torch.nn as nn
import math


class SinusoidalEmbedding1D(nn.Module):
    """
    Standard 1D sinusoidal positional embedding (linear).
    Used for the rho (log-radius) axis.
    """

    def __init__(self, num_pos_feats):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = 10000
        # This is a buffer, not a learnable parameter
        self.register_buffer('scale', torch.ones(1))

    def forward(self, max_len, device):
        # Create a positions tensor: [0, 1, 2, ..., max_len-1]
        pos = torch.arange(max_len, dtype=torch.float32, device=device)

        # Calculate the denominator for the PE formula
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        pos_emb = pos[:, None] / dim_t[None, :]
        pos_emb[:, 0::2] = pos_emb[:, 0::2].sin()
        pos_emb[:, 1::2] = pos_emb[:, 1::2].cos()

        return pos_emb  # Shape: [max_len, num_pos_feats]


class CyclicSinusoidalEmbedding1D(nn.Module):
    """
    Cyclic 1D sinusoidal positional embedding (Fourier features).
    Used for the theta (angle) axis.
    """

    def __init__(self, num_pos_feats):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        # Use frequencies that are powers of 2,
        # ensuring a mix of low and high frequencies
        # These are exponents, so freqs will be 2^0, 2^1, 2^2...
        self.register_buffer(
            'freqs',
            2.0 ** torch.arange(0, num_pos_feats // 2)
        )

    def forward(self, max_len, device):
        # Create normalized angular positions: [0, 2pi/W, 4pi/W, ..., 2pi*(W-1)/W]
        # This maps coordinates [0, ..., max_len-1] to the [0, 2pi) range
        pos = torch.arange(max_len, dtype=torch.float32, device=device)
        pos_norm = pos * (2.0 * math.pi / max_len)

        # pos_norm shape: [max_len]
        # freqs shape: [num_pos_feats / 2]

        # Outer product: [max_len, num_pos_feats / 2]
        pos_emb = pos_norm[:, None] * self.freqs[None, :]

        # Concatenate sin and cos components
        # [sin(k_1*theta), cos(k_1*theta), sin(k_2*theta), cos(k_2*theta), ...]
        pos_emb = torch.cat([pos_emb.sin(), pos_emb.cos()], dim=-1)

        return pos_emb  # Shape: [max_len, num_pos_feats]


class LogPolarPositionalEmbedding(nn.Module):
    """
    Combines linear (rho) and cyclic (theta) embeddings for log-polar images.

    Args:
        d_model (int): Total dimensionality of the model.
        max_shape (tuple): The expected (H, W) of the log-polar images.
                           H = rho_bins, W = theta_bins
    """

    def __init__(self, d_model: int, max_shape=(128, 128)):
        super().__init__()

        self.d_model = d_model
        self.max_shape = max_shape

        # Divide the d_model dimensions between rho and theta
        # We give d_model/2 to each axis
        self.d_model_half = d_model // 2

        # Ensure d_model is even
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, but got {d_model}")

        self.rho_embed = SinusoidalEmbedding1D(self.d_model_half)
        self.theta_embed = CyclicSinusoidalEmbedding1D(self.d_model_half)

        # Pre-compute embeddings
        self.register_buffer('pos_embed', self._build_embedding())

    def _build_embedding(self):
        max_h, max_w = self.max_shape
        device = self.theta_embed.freqs.device  # Get a buffer's device

        # 1. Get 1D embeddings
        # emb_rho shape: [max_h, d_model/2]
        # emb_theta shape: [max_w, d_model/2]
        emb_rho = self.rho_embed(max_h, device)
        emb_theta = self.theta_embed(max_w, device)

        # 2. Expand to 2D
        # [max_h, d_model/2] -> [max_h, 1, d_model/2] -> [max_h, max_w, d_model/2]
        emb_rho = emb_rho[:, None, :].repeat(1, max_w, 1)
        # [max_w, d_model/2] -> [1, max_w, d_model/2] -> [max_h, max_w, d_model/2]
        emb_theta = emb_theta[None, :, :].repeat(max_h, 1, 1)

        # 3. Concatenate to form the final 2D embedding
        # Shape: [max_h, max_w, d_model]
        pos_2d = torch.cat([emb_rho, emb_theta], dim=-1)

        # Re-order to [d_model, max_h, max_w] to match typical
        # (C, H, W) tensor layout for adding to features
        pos_2d = pos_2d.permute(2, 0, 1)

        # Add a batch dimension for broadcasting
        return pos_2d.unsqueeze(0)  # Shape: [1, d_model, max_h, max_w]

    def forward(self, features):
        """
        Adds the positional embedding to the input feature map.

        Args:
            features (torch.Tensor): Input features of shape [B, C, H, W]
                                      where C must equal d_model.
        Returns:
            torch.Tensor: Features with positional embeddings added.
        """
        B, C, H, W = features.shape

        # Add the pre-computed embedding (which broadcasts to batch size B)
        # We slice the pre-computed embedding to match the input H, W
        return features + self.pos_embed[:, :, :H, :W]