import torch
from torch import nn
import torchvision

import project_utils


class TargetGlimpseExtractor(nn.Module):
    """
    Given a clean object image with mask, produces glimpses from bottom, center, top (left, right, if specified)
    parts of the object
    """
    def __init__(
            self,
            sensor,
            top_center_bottom_only=True,
            with_masks=False,
            end_with_central_glimpse=False  # see the comment in 'def forward(...)'
    ):
        super().__init__()
        self.with_masks = with_masks
        self.top_center_bottom_only = top_center_bottom_only
        self.sensor = sensor
        self.end_with_central_glimpse = end_with_central_glimpse

    def forward(self, object_image, saliency_mask, **kwargs):
        batch_size = object_image.shape[0]

        boxes = torchvision.ops.masks_to_boxes(saliency_mask)
        center_locs = boxes[:, :2] + (boxes[:, 2:] - boxes[:, :2]) / 2

        all_extreme_locs = []
        for b in range(batch_size):
            extreme_locs = project_utils.get_mask_extreme_pixels(saliency_mask[b])
            if self.top_center_bottom_only:
                extreme_locs = torch.stack([extreme_locs['top'], extreme_locs['bottom']])
            else:
                extreme_locs = torch.stack([v for k, v in extreme_locs.items()])
            all_extreme_locs.append(extreme_locs)
        all_extreme_locs = torch.stack(all_extreme_locs)
        all_locs = torch.cat([center_locs[:, None], all_extreme_locs], dim=1)
        all_locs = project_utils.from_pixel_based_to_normalized(all_locs, saliency_mask.shape[-2:])

        if self.end_with_central_glimpse:
            # append the sequence with the central glimpse to match the training set-up where the model receives
            # 4 search target glimpses in total: top, central, bottom and a glimpse at a random location that
            # specifies the exact part of the object to be localized.
            # Hence, the appended central glimpse at test time mimics the last (4th) glimpse provided during training
            all_locs = torch.cat([all_locs, all_locs[:, :1]], dim=1)

        all_glimpses = []
        all_masks = []
        for l_idx in range(all_locs.shape[1]):
            all_glimpses.append(self.sensor(object_image, all_locs[:, l_idx])[:, 0])
            if self.with_masks:
                all_masks.append(self.sensor(saliency_mask[:, None], all_locs[:, l_idx])[:, 0])
        all_glimpses = torch.stack(all_glimpses, dim=1)

        if self.with_masks:
            all_masks = torch.stack(all_masks, dim=1)
            return all_glimpses, all_locs, all_masks
        else:
            return all_glimpses, all_locs, None
