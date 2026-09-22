import torch
import torch.nn as nn
import torch.nn.functional as F

from fine_glimpsing.basic_modules import BottomUpTopDownAttention


class FeatureCorrelator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scene, target_features, **kwargs):
        """
        :param scene: [B, H*W, D]
        :param target_features: [B, N_D, D]
        :param kwargs:
        :return:
        """
        search_map = (scene[..., None, :] * target_features[:, None]).mean([-1, -2])
        return search_map


class FineSearchMapGeneration(nn.Module):
    def __init__(
            self,
            scene_encoder: BottomUpTopDownAttention,
            target_encoder: BottomUpTopDownAttention,
            feature_comparison: FeatureCorrelator,
    ):
        super().__init__()

        self.feature_comparison = feature_comparison
        self.target_encoder = target_encoder
        self.scene_encoder = scene_encoder

    def forward(
            self,
            scene_glimpse,
            target_glimpses,
            precomputed_target_encoding=None,
    ):
        """
        :param scene_glimpse: [B, C, H_theta, W_rho]
        :param target_glimpses: [B, C, H_theta, W_rho] or [B, N_exmpl, C, H_theta, W_rho]
            or [B, N_exmpl, N_T, C, H_theta, W_rho] where N_T is the the number of glimpses for each example image
            of the search target
        :return: fine search map [B, H_theta, W_rho]

        Note that all input glimpses are of size [H_theta, W_rho] (for legacy reasons). However, search map generation
        works with transposed shape - this is why the glimpses are transposed at the beginning and the search map
        is transposed back to [H_theta, W_rho] to ensure compatibility with the rest of the CF-GAP system
        """
        if precomputed_target_encoding is not None:
            B, N_exmpl = precomputed_target_encoding.shape[:2]
        else:
            B, N_exmpl = target_glimpses.shape[:2]
            target_glimpses = target_glimpses.transpose(-1, -2)

        if scene_glimpse is not None:   # if scene is None only the target encoding will be computed
            scene_glimpse = scene_glimpse.transpose(-1, -2)

            scene_glimpse = self.scene_encoder(scene_glimpse, output_key='image_features', keep_grid_dims=True)
            enc_spatial_size = scene_glimpse.shape[1:3]
            scene_glimpse = scene_glimpse.flatten(1, 2)

        all_search_maps, all_target_encodings = list(), list()
        for i in range(N_exmpl):
            if precomputed_target_encoding is None:
                viewpoint_search_target_kp = target_glimpses[:, i]
                target_encoding = self.target_encoder(
                    viewpoint_search_target_kp, output_key='external_features', keep_grid_dims=True,
                    multi_glimpse=True
                )
            else:
                target_encoding = precomputed_target_encoding[:, i]
            all_target_encodings.append(target_encoding)

            if scene_glimpse is not None:
                search_map = self.feature_comparison(scene_glimpse, target_encoding)
                search_map = search_map.unflatten(1, enc_spatial_size)

                all_search_maps.append(search_map)
        if scene_glimpse is None:
            return torch.stack(all_target_encodings, 1)

        if N_exmpl == 1:
            search_map = all_search_maps[0]
        else:
            # if several examples of the search target are provided, average search maps generated for each of them
            search_map = torch.stack(all_search_maps, -1).mean(-1)

        return search_map.transpose(-1, -2)


class FineSearchMapGenerationWrapper(nn.Module):
    def __init__(
            self,
            backbone: FineSearchMapGeneration,
            every_n_example=1,
    ):
        super().__init__()
        self.every_n_example = every_n_example
        self.backbone = backbone

    def generate_target_encoding(self, search_target):
        """
        :param search_target: [B, N, glimpses per viewpoint, C, H, W]
        :return:
        """
        target_encoding = self.backbone(
            scene_glimpse=None, target_glimpses=search_target[:, 0::self.every_n_example, :, :3])
        return target_encoding

    def forward(self, scene, search_targets=None, target_encoding=None, **kwargs):
        assert not (search_targets is None and target_encoding is None)
        search_map = self.backbone(scene, target_glimpses=search_targets, precomputed_target_encoding=target_encoding)
        return search_map
