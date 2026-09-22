from typing import Literal

import torch
import torch.nn as nn

from fine_glimpsing.positional_embeddings import LogPolarPositionalEmbedding


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention module.

    This module allows for query, key, and value to come from different
    input tensors (x_q, x_k, x_v).
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate linear projections for Q, K, V
        # This is the key part for handling different sources.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor):
        """
        Forward pass.

        Args:
            x_q (torch.Tensor): Query tensor. Shape (B, N_q, C)
            x_k (torch.Tensor): Key tensor. Shape (B, N_k, C)
            x_v (torch.Tensor): Value tensor. Shape (B, N_v, C)
        """
        B_q, N_q, C = x_q.shape
        B_k, N_k, _ = x_k.shape
        B_v, N_v, _ = x_v.shape

        assert N_k == N_v, "Sequence lengths of K and V must be the same"

        # Project Q, K, V
        # (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = self.q_proj(x_q).reshape(B_q, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x_k).reshape(B_k, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x_v).reshape(B_v, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        # (B, H, N_q, D_h) @ (B, H, D_h, N_k) -> (B, H, N_q, N_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, H, N_q, N_k) @ (B, H, N_v, D_h) -> (B, H, N_q, D_h)
        # Note: N_k == N_v
        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# --- The ViT-style Block ---
class CrossAttentionBlock(nn.Module):
    """
    A ViT-style transformer block with cross-attention.

    This block implements the Pre-Normalization structure:
    x_q = x_q + DropPath(Attention(Norm(x_q), Norm(x_k), Norm(x_v)))
    x_q = x_q + DropPath(MLP(Norm(x_q)))

    The forward pass is flexible:
    - forward(x_q): Self-attention
    - forward(x_q, x_k): Cross-attention (K=V=x_k)
    - forward(x_q, x_k, x_v): Generalized cross-attention
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            # drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            share_norm='',
            skip_connection_pass_inp: Literal[None, 'q', 'k', 'v'] = 'q'
    ):
        super().__init__()

        # --- Attention Path ---
        # We need separate norms for Q, K, V as they can be different tensors
        self.skip_connection_pass_inp = skip_connection_pass_inp
        self.norm_q = norm_layer(dim)
        self.norm_k = self.norm_q if ('k' in share_norm and 'q' in share_norm) else norm_layer(dim)
        if 'v' in share_norm and 'k' in share_norm:
            self.norm_v = self.norm_k
        elif 'v' in share_norm and 'q' in share_norm:
            self.norm_v = self.norm_q
        else:
            self.norm_v = norm_layer(dim)

        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- MLP Path ---
        self.norm_mlp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop
        )
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x_q: torch.Tensor,
            x_k=None,
            x_v=None
    ) -> torch.Tensor:
        """
        Forward pass with flexible Q, K, V inputs.

        Args:
            x_q (torch.Tensor): Query tensor (B, N_q, C). This is the main
                                tensor that flows through the block.
            x_k (torch.Tensor, optional): Key tensor (B, N_k, C).
                                If None, x_k = x_q (self-attention).
            x_v (torch.Tensor, optional): Value tensor (B, N_v, C).
                                If None, x_v = x_k.

        Returns:
            torch.Tensor: Output tensor, shape (B, N_q, C)
        """
        # --- Handle default cases for K and V ---
        # 1. If x_k is None, x_k = x_q (self-attention)
        x_k = x_q if x_k is None else x_k

        # 2. If x_v is None, x_v = x_k (standard self- or cross-attention)
        x_v = x_k if x_v is None else x_v

        # --- Attention Path ---
        # x_q = x_q + DropPath(Attention(Norm(x_q), Norm(x_k), Norm(x_v)))
        attn_out = self.attn(
            self.norm_q(x_q),
            self.norm_k(x_k),
            self.norm_v(x_v)
        )
        x_q = x_q + attn_out

        # --- MLP Path ---
        # x_q = x_q + DropPath(MLP(Norm(x_q)))
        mlp_out = self.mlp(self.norm_mlp(x_q))

        if self.skip_connection_pass_inp is None:
            x_q = mlp_out
        else:
            skip_pass_input = {
                'q': x_q,
                'k': x_k,
                'v': x_v,
            }
            x_q = skip_pass_input[self.skip_connection_pass_inp] + mlp_out

        return x_q


class BottomUpTopDownAttention(nn.Module):
    """
    Implements encoders and decoders. Terms 'external_embeddings' and 'external_features' are used interchangeably
    """
    def __init__(
            self,
            n_feats,
            feat_dim,
            bu_attn,
            td_attn,
            patch_encoder=None,
            feat_self_attention=None,
            use_external_features_td_values=False,
            grid_size=None,
            max_feat_norm=None,
            projection=None,
            use_instance_norm=False,
            use_lnorm_external_features=False,
            concat_pos_embeddings=False,
    ):
        super().__init__()

        self.concat_pos_embeddings = concat_pos_embeddings
        self.projection = projection
        self.feat_self_attention = feat_self_attention
        self.patch_encoder = patch_encoder
        self.bu_attn = bu_attn
        self.td_attn = td_attn
        self.grid_size = grid_size

        if use_external_features_td_values:
            self.external_features_td_values = nn.Embedding(n_feats, feat_dim)
        else:
            self.external_features_td_values = None

        self.learnable_embeddings = nn.Embedding(n_feats, feat_dim, max_norm=max_feat_norm)
        self.n_feats = n_feats

        pos_embedding_dim = feat_dim if self.patch_encoder is None else self.patch_encoder.out_channels
        pos_embedding = LogPolarPositionalEmbedding(pos_embedding_dim, self.grid_size).pos_embed
        self.pos_embedding = nn.Parameter(pos_embedding.flatten(-2).permute(0, 2, 1))  # [1, rho*theta, D],

        if use_lnorm_external_features:
            self.ext_layer_norm = nn.LayerNorm(feat_dim, elementwise_affine=False)
        else:
            self.ext_layer_norm = None

        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm2d(self.grid_size)
        else:
            self.instance_norm = None

    def _forward_single_glimpse(
            self, glimpse, learnable_embeddings=None, keep_grid_dims=False, output_key=None):
        """
        :param glimpse: or spatial features, e.g. glimpse encoding
        :param learnable_embeddings: if None, only non-spatial embedding is produced (e.g. target encoding)
        :param keep_grid_dims:
        :param output_key:
        :return:
        """
        B = glimpse.shape[0]

        if learnable_embeddings is None:
            learnable_embeddings = self.learnable_embeddings(
                torch.arange(self.n_feats, device=glimpse.device)).unsqueeze(0).expand(B, -1, -1)

        if self.patch_encoder is not None:
            feature_grid = self.patch_encoder(glimpse).flatten(2).transpose(1, 2)
        else:
            feature_grid = glimpse
        # 'feature_grid' [B, HxW, D] is always flattened so that all patches are treated as a sequence of tokens

        feature_grid = feature_grid + self.pos_embedding

        if self.ext_layer_norm is not None:
            learnable_embeddings = self.ext_layer_norm(learnable_embeddings)

        # --- BU cross attention
        v = feature_grid
        learnable_embeddings = self.bu_attn(x_q=learnable_embeddings, x_k=feature_grid, x_v=v)    # [B, N_E, D]

        if self.feat_self_attention is not None:
            learnable_embeddings = self.feat_self_attention(learnable_embeddings)     # [B, N_E, D]
        if output_key == 'external_features':
            return dict(external_features=learnable_embeddings)

        # --- TD cross attention
        if self.external_features_td_values is not None:
            v = self.external_features_td_values(
                torch.arange(self.n_feats, device=glimpse.device)).unsqueeze(0).expand(B, -1, -1)
        else:
            v = learnable_embeddings
        feature_grid = self.td_attn(x_q=feature_grid, x_k=learnable_embeddings, x_v=v)       # [B, HxW, D]

        if self.instance_norm is not None:
            feature_grid = self.instance_norm(feature_grid.unflatten(1, self.grid_size)).flatten(1, 2)

        if self.projection is not None:
            readout_image_feats = self.projection(feature_grid)
            if readout_image_feats.shape[-1] == 1:  # if the readout is for the 1-dim search map
                readout_image_feats = readout_image_feats[..., 0]
        else:
            readout_image_feats = None

        if keep_grid_dims:
            feature_grid = feature_grid.unflatten(1, self.grid_size)
            if readout_image_feats is not None:
                readout_image_feats = readout_image_feats.unflatten(1, self.grid_size)

        return dict(
            image_features=feature_grid, readout_image_feats=readout_image_feats, external_features=learnable_embeddings)

    def forward(
            self,
            glimpse,
            external_embeddings=None,
            output_key: Literal['image_features', 'external_features'] = None,
            keep_grid_dims=False, multi_glimpse=False,
    ):
        """
        :param glimpse:  [B, N_T, C, H_rho, W_theta] for search target glimpses (N_T - number of glimpses);
                         [B, C, H_rho, W_theta] for scene glimpse
        :param external_embeddings: None for scene and search target encoders, for search map decoder [B, N_E, D]
                                    N_E is the number of vectors in the embedding (e.g. target encoding)
        :param output_key: 'image_features' - only spatial encoding (e.g. scene encoding) will be returned
                           'external_features' - only embeddings (e.g. target encoding) will be returned
                           None - everything will be returned
        :param keep_grid_dims:
        :param multi_glimpse:
        :return:
        """
        if not multi_glimpse:
            # used to produce an encoding for a single, e.g. scene glimpse
            out = self._forward_single_glimpse(
                glimpse, learnable_embeddings=external_embeddings, keep_grid_dims=keep_grid_dims, output_key=output_key)
        else:
            for glimpse_idx in range(glimpse.shape[1]):
                out = self._forward_single_glimpse(
                    glimpse[:, glimpse_idx],
                    learnable_embeddings=external_embeddings, keep_grid_dims=keep_grid_dims, output_key=output_key)
                external_embeddings = out['external_features']

        if output_key is None:
            return out
        else:
            return out[output_key]
