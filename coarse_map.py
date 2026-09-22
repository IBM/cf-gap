from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as VF
import numpy as np

from project_utils import normalize, min_max_normalize


class CoarseSearchMapGeneration(nn.Module):
    def __init__(
            self,
            backbone_cnn,
            norm_type_feature_maps: Literal['l2', 'softmax', None] = None,
            norm_type_target: Literal['l2', 'softmax', None] = None,
            resize_scene=None,
    ):
        super().__init__()
        self.norm_type_target = norm_type_target
        self.norm_type_feature_maps = norm_type_feature_maps
        self.backbone_cnn = backbone_cnn
        self.resize_scene = resize_scene

    def compute_target_features(self, search_target_examples):
        if search_target_examples.shape[2] == 4:
            search_target_masks = search_target_examples[:, :, -1:]
            search_target_examples = search_target_examples[:, :, :3]
        else:
            search_target_masks = None

        target_features = self.backbone_cnn(
            search_target_examples.flatten(0, 1)).unflatten(0, search_target_examples.shape[:2])

        target_features = normalize(target_features, self.norm_type_feature_maps, 1)

        if search_target_masks is not None:
            search_target_masks = VF.resize(
                search_target_masks.flatten(0, 1), target_features.shape[-2:],
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            ).unflatten(0, search_target_masks.shape[:2])
            target_features = target_features * search_target_masks

        target_features_avg = target_features.mean([1, -1, -2])
        target_features_avg = normalize(target_features_avg, self.norm_type_target, -1)

        return target_features_avg

    def forward(self, scene, search_target_examples, return_all=False):
        if self.resize_scene is not None:
            scene = F.interpolate(scene, size=self.resize_scene, mode='bilinear')

        target_features_hist = self.compute_target_features(search_target_examples)
        if scene.shape[1] == 3:
            scene_features = self.backbone_cnn(scene)
        else:
            # scene was already preprocessed by some common backbone
            scene_features = scene

        scene_features_normalized = normalize(scene_features, self.norm_type_feature_maps, 1)

        search_map = (target_features_hist[..., None, None] * scene_features_normalized).sum(1)

        if return_all:
            return search_map, dict(scene_feature_map=scene_features)

        return search_map


class IoRMasker(nn.Module):
    def __init__(self, masking_space_size, eps):
        """
        :param masking_space_size: [H, W] of the spatial mask
        :param eps: size of the masking "spot", smaller 'eps' corresponds to smaller radius
        """
        super().__init__()

        if not isinstance(masking_space_size, list) and not isinstance(masking_space_size, torch.Size):
            masking_space_size = [masking_space_size, masking_space_size]

        ranges = [np.arange(res) for res in masking_space_size]
        grid = np.meshgrid(*ranges, sparse=False, indexing="xy")
        grid = np.stack(grid, axis=-1).astype(np.float32)
        self.register_buffer('grid', torch.from_numpy(grid).swapaxes(0, 1).flip(-1).unsqueeze(0).half())
        self.grid_size = self.grid.shape[1: 3]

        self.masking_space_size = masking_space_size
        self.eps = eps

    @torch.no_grad()
    def forward(self, locs, scales=None, eps=None):
        if eps is None:
            eps = self.eps
        if scales is None:
            scales = torch.ones(1, device=locs.device)

        # assert (locs.abs() <= 1.).all()
        locs = torch.tensor(self.masking_space_size, device=locs.device).flip(0) * (1 + locs) / 2
        locs = locs.half()

        var = eps
        sigma = np.sqrt(var)

        # --- compute RBF kernel
        if len(locs.shape) == 2:
            distances = (self.grid - locs[:, None, None]).norm(p=2, dim=-1).square()
        else:
            distances = (self.grid[:, :, :, None, None] - locs[:, None, None]).norm(p=2, dim=-1).square().swapaxes(-1, -2)
            # [B, H, W, L, scales]

        mask = torch.exp(-distances / (2 * var * scales.expand_as(distances)))
        # [B, H, W, L, scales]      (L is the total number of locations in 'locs')
        mask /= sigma * scales.expand_as(mask).sqrt() * np.sqrt(2 * np.pi)

        if len(locs.shape) != 2:
            mask = mask.permute(0, 4, 3, 1, 2)  # final shape [B, scales, L, H, W]

        mask = min_max_normalize(mask, -2)
        return mask.half()
