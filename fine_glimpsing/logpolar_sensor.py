import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import warnings

import project_utils


class LogPolarSensor(nn.Module):
    def __init__(
            self,
            glimpse_size, radius,
            skew=1.
    ):
        super().__init__()
        self.skew = skew
        self.radius = radius

        if glimpse_size[1] == -1:
            self.glimpse_size = (glimpse_size[0], int(glimpse_size[0] * np.pi))
        else:
            self.glimpse_size = glimpse_size

        rho_size, theta_size = self.glimpse_size

        # Create the base grid in log-polar coordinates
        theta = torch.linspace(0, 2 * math.pi, theta_size)

        rho_log = torch.linspace(0, math.log(self.radius), rho_size).exp()
        rho_lin = torch.linspace(1, self.radius, rho_size)
        rho = rho_log * self.skew + rho_lin * (1 - self.skew)

        rho_grid, theta_grid = torch.meshgrid(rho, theta, indexing="ij")

        # Convert log-polar coordinates to Cartesian coordinates
        # This is the inverse transformation
        x_base = rho_grid * torch.cos(theta_grid)
        y_base = rho_grid * torch.sin(theta_grid)

        self.register_buffer("rho_grid", rho_grid.clone())
        self.register_buffer("theta_grid", theta_grid.clone())
        self.register_buffer("xy_logpolar_grid_base", torch.stack([x_base, y_base], -1).clone())

    def cartesian_to_logpolar(self, cartesian_pixels, centers):
        """
        Converts a batch of cartesian pixel locations to their corresponding log-polar pixel locations.

        Args:
            transformer (LogPolarTransform): An instance of the LogPolarTransform class.
            cartesian_pixels (torch.Tensor): A tensor of (x, y) coordinates of shape (B, 2),
                                             where B is the batch size.
            centers (torch.Tensor): A tensor of (cx, cy) center coordinates of shape (B, 2).

        Returns:
            torch.Tensor: A tensor of corresponding (rho_idx, theta_idx) coordinates of shape (B, 2).
        """
        device = self.rho_grid.device
        cartesian_pixels = cartesian_pixels.to(device)
        centers = centers.to(device)

        # 1. Shift coordinates to be relative to the center (works element-wise for the whole batch)
        delta = cartesian_pixels - centers
        dx, dy = delta[:, 0], delta[:, 1]

        # 2. Convert Cartesian coordinates to polar coordinates for the entire batch
        r = torch.hypot(dx, dy)
        theta = torch.atan2(dy, dx)

        # Adjust theta to be in the [0, 2*pi] range for the entire batch
        theta = torch.where(theta < 0, theta + 2 * math.pi, theta)

        # 3. Find the closest indices in the log-polar grid using broadcasting
        rho_vector = self.rho_grid[:, 0]  # Shape: (rho_size,)
        theta_vector = self.theta_grid[0, :]  # Shape: (theta_size,)

        # For each radius in the batch, find the closest index in rho_vector
        # Unsqueeze to enable broadcasting: (rho_size, 1) vs (B,) -> (rho_size, B)
        rho_diffs = torch.abs(rho_vector.unsqueeze(1) - r)
        rho_indices = torch.argmin(rho_diffs, dim=0)

        # Do the same for the angle
        # Unsqueeze to enable broadcasting: (theta_size, 1) vs (B,) -> (theta_size, B)
        theta_diffs = torch.abs(theta_vector.unsqueeze(1) - theta)
        theta_indices = torch.argmin(theta_diffs, dim=0)

        # Stack results into a (B, 2) tensor
        return torch.stack([rho_indices, theta_indices], dim=1)

    def forward(self, image, location, *args, **kwargs):
        """
        Performs a log-polar transformation of a batch of images.

        Args:
            image (torch.Tensor): The input image tensor of shape (B, C, H, W).
            location (torch.Tensor): The center (x, y) of the transformation of shape (B, 2).
                                    Normalized in range [-1, 1]

        Returns:
            torch.Tensor: The log-polar transformed image.
        """
        if image.dim() != 4:
            raise ValueError("Input image must be a 4D tensor (B, C, H, W)")

        batch_size, _, H, W = image.shape
        device = image.device

        location = project_utils.from_normalized_to_pixel_based(location, image.shape)

        # ...but now we add the specific center for each image in the batch.
        # We reshape centers and the base coordinates for broadcasting.
        # centers shape: (B, 2) -> (B, 1, 1, 2)
        # x_base/y_base shape: (H_out, W_out) -> (1, H_out, W_out)
        location = location.to(device).view(batch_size, 1, 1, 2)
        x_centers = location[..., 0]
        y_centers = location[..., 1]

        x_base, y_base = self.xy_logpolar_grid_base[..., 0], self.xy_logpolar_grid_base[..., 1]

        x = x_base.unsqueeze(0) + x_centers
        y = y_base.unsqueeze(0) + y_centers

        # Normalize the Cartesian coordinates to the range [-1, 1] for grid_sample
        # The grid_sample function expects coordinates in the range [-1, 1],
        # where (-1, -1) is the top-left corner and (1, 1) is the bottom-right.
        normalized_x = (2 * x / (W - 1)) - 1
        normalized_y = (2 * y / (H - 1)) - 1

        # Stack to create the final grid for each image in the batch
        # The grid will have the shape (B, H_out, W_out, 2)
        grid = torch.stack((normalized_x, normalized_y), dim=-1)

        # Sample the input image using the generated grid
        transformed_image = F.grid_sample(
            image, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )

        return transformed_image.swapaxes(-1, -2).unsqueeze(1)

    def calculate_centroid(
            self,
            weight_map,
            normalized=False, eps_theta=0.5, eps_rho=10, is_rho_centroid_logpolar=True,
            return_sin_cos_centroid=False,
    ):
        """
        :param weight_map: search map, shape [B, H, W]
        :param normalized: whether to normalize map to make all its values sum to 1
        :return: centroid: (x, y) coordinates in cartesian space with the origin of log-polar
                        transformation
        :return: centroid_idx: (rho_idx, theta_idx) with in the image array [C, H_theta, W_rho]
                (note the flipped order of image shape and returned (rho_idx, theta_idx))
        """
        if not normalized:
            weight_map = weight_map / (weight_map.sum([1, 2], keepdims=True) + 1e-6)

        if not torch.allclose(weight_map.flatten(1).sum(1), torch.ones_like(weight_map.flatten(1).sum(1)), rtol=1e-3):
            warnings.warn("Some of the normalized maps are not zeros...")

        # --- Theta Centroid (Circular Mean) ---
        # Convert angles to vectors, find the average vector, then convert back to angle
        weighted_cos_theta = (weight_map.swapaxes(-1, -2) * torch.cos(self.theta_grid)).sum(dim=(-2, -1))
        weighted_sin_theta = (weight_map.swapaxes(-1, -2) * torch.sin(self.theta_grid)).sum(dim=(-2, -1))
        theta_centroid_idx = project_utils.sin_cos_to_theta(
            weighted_sin_theta, weighted_cos_theta, self.xy_logpolar_grid_base.shape[1] )

        # --- calculate rho centroid
        if is_rho_centroid_logpolar:
            centroid = (weight_map.swapaxes(-1, -2)[..., None] * self.xy_logpolar_grid_base[None])
            centroid = centroid.flatten(1, 2).sum(1)

            rho_centroid = torch.norm(centroid, p=2, dim=-1)
            rho_centroid_idx = torch.log(rho_centroid) * (self.xy_logpolar_grid_base.shape[0] - 1) / math.log(self.radius)
            if self.skew != 1.:
                rho_centroid_idx_lin = rho_centroid * (self.xy_logpolar_grid_base.shape[0] - 1) / self.radius
                rho_centroid_idx = self.skew * rho_centroid_idx + (1 - self.skew) * rho_centroid_idx_lin
        else:
            uniform_rho_grid = torch.linspace(0., 1., self.rho_grid.shape[0], device=self.rho_grid.device)
            uniform_rho_grid = uniform_rho_grid[None, :, None].expand(weight_map.shape[0], -1, self.rho_grid.shape[1])
            rho_centroid_idx = (weight_map.swapaxes(-1, -2) * uniform_rho_grid).sum([1, 2]) * self.rho_grid.shape[0]

        rho_centroid_idx = rho_centroid_idx.clip(0, self.xy_logpolar_grid_base.shape[0]-1)

        centroid_idx = torch.stack([rho_centroid_idx, theta_centroid_idx], -1)
        centroid = self.xy_logpolar_grid_base[rho_centroid_idx.int(), theta_centroid_idx.int()]

        return centroid, centroid_idx
