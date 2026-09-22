import torch
from torch import nn
import torch.nn.functional as F

from fine_glimpsing.fine_gap import FineGAP
import project_utils


class CoarseToFineGAP(nn.Module):
    def __init__(
            self,
            coarse_map_extractor,
            n_coarse_glimpses,
            fine_glimpsing: FineGAP,
            ior_masker,
            search_target_glimpses_extraction,
            ior_coarse_glimpses_only=False,
    ):
        super().__init__()

        self.search_target_glimpses_extraction = search_target_glimpses_extraction
        self.ior_coarse_glimpses_only = ior_coarse_glimpses_only
        self.coarse_map_extractor = coarse_map_extractor
        self.n_coarse_glimpses = n_coarse_glimpses
        self.fine_glimpsing = fine_glimpsing
        self.ior_masker = ior_masker

    def forward(self, scene, search_target, return_all=False, **kwargs):
        """
        :param scene: [B, C, H, W], original scene image where the search target has to be located
        :param search_target: [B, N_exmpl, 4, H, W], example images of the search target object, each example provides
                            a different viewpoint. The last channel dimension contains a segmentation mask
        :param return_all:
        :return:
        """
        try:
            self.fine_glimpsing.downstream_architecture.initialize(search_targets_original=search_target)
        except AttributeError as e:
            print(e)
            print("Fine GAP has no downstream architecture")

        coarse_priority_map, _ = self.coarse_map_extractor(scene, search_target, return_all=True)
        coarse_priority_map = project_utils.min_max_normalize(coarse_priority_map)

        # extract glimpses from search target examples at top, center, bottom locations
        search_target_glimpses, _, _ = self.search_target_glimpses_extraction(
            search_target.flatten(0, 1)[:, :3], search_target.flatten(0, 1)[:, 3]
        )
        search_target_glimpses = search_target_glimpses.unflatten(0, search_target.shape[:2])
        target_encoding = self.fine_glimpsing.search_map_generator.generate_target_encoding(search_target_glimpses)

        all_loc_hist = []
        acc_ior_mask = None

        best_verification_score = -torch.inf
        best_matching_output = dict(pred_obj_center_loc=None, bounding_boxes=None)

        for t in range(self.n_coarse_glimpses):
            coarse_glimpse_location, _ = project_utils.get_wta_loc_from_map(coarse_priority_map)

            fine_glimpse_locations, matching_output = self.fine_glimpsing(
                scene, search_target_glimpses,
                coarse_glimpse_location,
                target_encoding=target_encoding,
                return_all=return_all,
            )

            if matching_output:
                if matching_output['score'] > best_verification_score:
                    best_verification_score = matching_output['score']
                    best_matching_output = dict()
                    for k, v in matching_output.items():
                        best_matching_output[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v

            # update the coarse search map with IoR
            new_ior_mask = self.get_ior_mask(coarse_glimpse_location, fine_glimpse_locations)
            acc_ior_mask = new_ior_mask if acc_ior_mask is None else acc_ior_mask * new_ior_mask
            coarse_priority_map = acc_ior_mask * F.interpolate(
                coarse_priority_map[:, None], acc_ior_mask.shape[-2:], mode='bilinear')[:, 0]

            # for debugging and analysis
            all_loc_hist.append(torch.cat([coarse_glimpse_location[:, None], fine_glimpse_locations], dim=1).cpu())
            if matching_output is not None and matching_output['success']:
                break

        # reset downstream architecture, if possible
        try:
            self.fine_glimpsing.verification.reset()
        except AttributeError:
            pass

        aux_output_dict = dict(
            loc_hist=torch.stack(all_loc_hist, dim=1),
        )

        return (best_matching_output['pred_obj_center_loc'],
                best_matching_output['bounding_boxes'],
                aux_output_dict)

    def get_ior_mask(
            self,
            coarse_glimpse_location,
            fine_glimpse_locations,
    ):
        if self.ior_coarse_glimpses_only:
            all_locations = coarse_glimpse_location[:, None]
        else:
            all_locations = torch.cat([coarse_glimpse_location[:, None], fine_glimpse_locations], dim=1)
        all_masks = self.ior_masker(all_locations.flatten(0, 1)).unflatten(0, all_locations.shape[:2])
        final_ior_mask = (1 - all_masks).prod(1)

        return final_ior_mask
