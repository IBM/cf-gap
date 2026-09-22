from typing import Literal
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

import project_utils


class FineGAP(nn.Module):
    def __init__(
            self,
            sensor,
            search_map_generator,
            n_glimpses: int = None,
            downstream_architecture=None,
            temperature=1.,
            is_centroid_log_polar=True,
            location_selection_type: Literal['centroid', 'wta'] = 'centroid',
    ):
        super().__init__()
        self.location_selection_type = location_selection_type
        self.downstream_architecture = downstream_architecture
        self.temperature = temperature
        self.is_centroid_log_polar = is_centroid_log_polar
        self.n_glimpses = n_glimpses
        self.search_map_generator = search_map_generator
        self.sensor = sensor

    def glimpsing_step(self, target_encoding, glimpse_scene):
        out = self.search_map_generator(
            glimpse_scene,
            search_targets=None,    # not needed, as the target encoding is pre-computed
            target_encoding=target_encoding,
        )
        if isinstance(out, tuple):
            search_map, aux_out = out
        else:
            search_map, aux_out = out, None

        if self.location_selection_type == 'wta':
            best_rho_theta_inds_logpolar, _ = project_utils.get_wta_loc_from_map(search_map)
            best_rho_theta_inds_logpolar = project_utils.from_normalized_to_pixel_based(
                best_rho_theta_inds_logpolar, glimpse_scene.shape[-2:])
        elif self.location_selection_type == 'centroid':
            search_map = F.interpolate(search_map[:, None], glimpse_scene.shape[-2:])[:, 0]
            _, best_rho_theta_inds_logpolar = self.sensor.calculate_centroid(
                project_utils.spatial_softmax(self.temperature * search_map, spatial_dim_start=1),
                is_rho_centroid_logpolar=self.is_centroid_log_polar, normalized=True,)
        else:
            raise NotImplementedError
        best_rho_theta_inds_logpolar = best_rho_theta_inds_logpolar.int()

        # convert location from log-polar space to xy cartesian space w.r.t. the current glimpse loc
        best_xy_locs_cartesian = self.sensor.xy_logpolar_grid_base[
            best_rho_theta_inds_logpolar[..., 0], best_rho_theta_inds_logpolar[..., 1]]

        return search_map, best_xy_locs_cartesian, best_rho_theta_inds_logpolar

    def forward(
            self,
            scene,
            search_target,
            initial_glimpse_loc,    # coarse glimpse location
            target_encoding=None,
            custom_num_glimpses=None,  # mainly for debugging
            **kwargs,
        ):

        if target_encoding is None:
            target_encoding = self.search_map_generator.generate_target_encoding(search_target)

        current_glimpse_loc = initial_glimpse_loc
        current_glimpse = self.sensor(scene, current_glimpse_loc)[:, 0]

        loc_hist = [current_glimpse_loc]

        # --- glimpsing loop
        num_glimpses = custom_num_glimpses if custom_num_glimpses is not None else self.n_glimpses
        for t in range(num_glimpses):
            search_map, centroid, centroid_log_polar = self.glimpsing_step(
                target_encoding, current_glimpse)

            # --- update glimpse location and glimpse scene
            current_glimpse_loc = project_utils.from_normalized_to_pixel_based(
                current_glimpse_loc, scene.shape) + centroid
            current_glimpse_loc = project_utils.from_pixel_based_to_normalized(current_glimpse_loc, scene.shape)
            current_glimpse_loc = current_glimpse_loc.clip(-1, 1)

            current_glimpse = self.sensor(scene, current_glimpse_loc)[:, 0]

            loc_hist.append(current_glimpse_loc)

        loc_hist = torch.stack(loc_hist, dim=1)

        if self.downstream_architecture:
            matching_output = self.downstream_architecture(scene, loc_hist)
        else:
            matching_output = None

        fine_glimpse_locs = loc_hist[:, 1:]     # the first location in 'loc_hist' is the initial coarse glimpse loc
        return fine_glimpse_locs, matching_output
