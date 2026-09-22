from typing import Literal

import numpy as np
import torchvision.ops
import torchvision.transforms.functional as VF
import torch
import random
import torch.nn.functional as F
import math
import kornia
import torch.nn as nn


def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def from_normalized_to_pixel_based(location, image_shape):
    """
    :param location: expected shape [B, ..., 2], locations (x, y) have to be normalized within range [-1, 1]
    :param image_shape: [B, C, H, W] - only the last two elements are used
    :return:
    """
    return (torch.tensor(image_shape[-2:], device=location.device).flip(0) * (location + 1) / 2).int()


def from_pixel_based_to_normalized(location, image_shape):
    """
    :param location: expected shape [B, ..., 2], locations (x, y) to be in the range [0, H-1]x[0, W-1]
    :param image_shape: [B, C, H, W] - only the last two elements are used
    :return:
    """
    return 2 * (location.float() / torch.tensor(image_shape[-2:], device=location.device).flip(0)) - 1


def min_max_normalize(map_2D, spatial_dim_start=1):
    max_v = map_2D.flatten(spatial_dim_start).max(-1).values[..., None, None]
    min_v = map_2D.flatten(spatial_dim_start).min(-1).values[..., None, None]

    return (map_2D - min_v) / (max_v - min_v + 1e-6)


def get_wta_loc_from_map(maps, normalized=True, top_k=None):
    # maps.shape: [B, ..., H, W]
    if top_k is None:
        best_loc = maps.flatten(1).argmax(-1)
    else:
        best_loc = maps.flatten(1).topk(top_k, -1).indices

    best_loc_unraveled = torch.unravel_index(best_loc, maps.shape[1:])

    best_xy_loc = torch.stack(best_loc_unraveled[-2:], -1).flip(-1)

    if normalized:
        best_xy_loc = from_pixel_based_to_normalized(best_xy_loc, maps.shape[-2:])

    if len(best_loc_unraveled) > 2:
        aux_loc = torch.stack(best_loc_unraveled[:-2], -1)
    else:
        aux_loc = None

    return best_xy_loc, aux_loc


def spatial_softmax(_map, spatial_dim_start):
    return _map.flatten(spatial_dim_start).softmax(-1).unflatten(-1, _map.shape[spatial_dim_start:])


def sin_cos_to_theta(sin_coords, cos_coords, theta_size):
    theta_centroid = torch.atan2(sin_coords, cos_coords)
    theta_centroid = (theta_centroid + 2 * math.pi) % (2 * math.pi)
    theta_centroid_idx = theta_centroid * (theta_size - 1) / (2 * math.pi)

    return theta_centroid_idx


def normalize(inp, norm_type: Literal['l2', 'softmax', None], dim):
    if norm_type == 'l2':
        return F.normalize(inp, dim=dim)
    elif norm_type == 'softmax':
        return F.softmax(inp, dim=dim)
    elif norm_type is None:
        return inp
    else:
        raise NotImplementedError


def get_mask_extreme_pixels(mask: torch.Tensor):
    """
    Finds the 2D coordinates of the most top, bottom, left, and right
    pixels in the mask.

    Args:
        mask (torch.Tensor): A 2D tensor of shape (H, W) with boolean or 0/1 values.

    Returns:
        dict: A dictionary with 'top', 'bottom', 'left', 'right' coordinates
              as [row, col]
    """
    # Find all non-zero coordinates.
    # coords shape is (N, 2), where N is num of 'True' pixels
    # coords[:, 0] is rows (y), coords[:, 1] is cols (x)
    coords = torch.nonzero(mask.bool(), as_tuple=False)

    rows = coords[:, 0]
    cols = coords[:, 1]

    # Find the *index* (in the coords tensor) of the min/max row
    top_pixel_idx = torch.argmin(rows)
    bottom_pixel_idx = torch.argmax(rows)

    # Find the *index* (in the coords tensor) of the min/max col
    left_pixel_idx = torch.argmin(cols)
    right_pixel_idx = torch.argmax(cols)

    # Get the (row, col) coordinates at these indices
    top_coord = coords[top_pixel_idx]
    bottom_coord = coords[bottom_pixel_idx]
    left_coord = coords[left_pixel_idx]
    right_coord = coords[right_pixel_idx]

    return {
        "top": top_coord.flip(-1),
        "bottom": bottom_coord.flip(-1),
        "left": left_coord.flip(-1),
        "right": right_coord.flip(-1),
    }


def pad_image_to_square(image: torch.Tensor, pad_value: float = 0) -> torch.Tensor:
    """
    Pads a given image tensor to a square shape.

    The function takes a PyTorch tensor, which can be in the format (C, H, W)
    or (B, C, H, W), and pads the smaller dimension (height or width)
    to match the larger dimension, resulting in a square image. The original
    image is centered within the new padded image.

    Args:
        image (torch.Tensor): The input image tensor.
                              Shape can be (C, H, W) or (B, C, H, W).
        pad_value (float): The value to use for padding. Defaults to 0 (black).

    Returns:
        torch.Tensor: The padded, square image tensor.
    """
    # Get the shape of the image.
    # We assume the height and width are the last two dimensions.
    _, _, h, w = image.shape if image.dim() == 4 else (None, *image.shape)

    if h == w:
        # The image is already square, no padding needed.
        return image

    # Determine the larger dimension
    max_dim = max(h, w)

    # Calculate the total padding needed
    pad_h = max_dim - h
    pad_w = max_dim - w

    # Calculate padding for each side (left, right, top, bottom)
    # We use integer division and subtraction to handle odd padding amounts
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # The 'pad' argument for F.pad is in the order:
    # (padding_left, padding_right, padding_top, padding_bottom)
    # This order is for the last two dimensions of the tensor.
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    # Pad the image
    padded_image = F.pad(image, padding, mode='constant', value=pad_value)

    return padded_image


def apply_model_on_bb_regions(model, images, bbs, resize, pad_to_square=False):
    object_bounding_boxes = bbs.int()
    all_crops = []

    for b in range(images.shape[0]):
        crop = VF.crop(
            images[b:b+1, :3],
            object_bounding_boxes[b, 1], object_bounding_boxes[b, 0],
            object_bounding_boxes[b, 3] - object_bounding_boxes[b, 1],
            object_bounding_boxes[b, 2] - object_bounding_boxes[b, 0],
        )
        if pad_to_square:
            crop = pad_image_to_square(crop)

        all_crops.append(VF.resize(crop, resize))

    all_crops = torch.cat(all_crops)
    return model(all_crops)


def auto_adjust_contrast(image):
    # expected shape [B, C, H, W], all values are in [0,1]
    equalized_scene = kornia.color.rgb_to_lab(image)

    l_channel = VF.autocontrast(equalized_scene[:, :1]/100) * 100

    equalized_scene = torch.cat([l_channel, equalized_scene[:, 1:]], 1)
    equalized_scene = kornia.color.lab_to_rgb(equalized_scene)
    return equalized_scene


def noisy_masks_to_bboxes(
    masks: torch.Tensor,
    kernel_size: int = 5
):
    """
    Converts a batch of noisy binary masks to bounding boxes.

    This function first applies a morphological opening (erosion followed by
    dilation) to remove small noise artifacts from the masks. Then, it calculates
    the bounding box of the largest remaining component.

    Args:
        masks (torch.Tensor): A batch of binary masks of shape (B, H, W),
                              where B is the batch size, H is the height, and
                              W is the width. The tensor should be of a float
                              or long type with values 0 or 1.
        kernel_size (int): The size of the square kernel used for the
                           morphological opening. This should be an odd
                           integer. A larger kernel will remove larger
                           noise elements.
    """
    if masks.ndim != 3:
        raise ValueError("Input masks must be a 3D tensor of shape (B, H, W)")

    B, H, W = masks.shape
    device = masks.device

    # Ensure masks are float for pooling operations and add a channel dimension
    masks_float = masks.clone().float().unsqueeze(1).to(device) # Shape: (B, 1, H, W)

    # 1. Erosion: Shrinks white regions, small noise islands disappear.
    # We simulate min_pool2d with a negative max_pool2d, a common trick.
    padding = kernel_size // 2
    eroded_masks = -F.max_pool2d(
        -masks_float,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    # 2. Dilation: Expands the remaining white regions back to their original size.
    cleaned_masks = F.max_pool2d(
        eroded_masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    # Binarize the result and remove the channel dimension
    cleaned_masks = (cleaned_masks > 0).squeeze(1).int() # Shape: (B, H, W)
    bounding_boxes = torchvision.ops.masks_to_boxes(cleaned_masks)

    return bounding_boxes, cleaned_masks




