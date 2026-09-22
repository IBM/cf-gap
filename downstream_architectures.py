import os

import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from torchvision.ops import box_convert

import project_utils as general_utils


class DownstreamArch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scene, glimpse_locs, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass

    def initialize(self, search_targets_original):
        """
        Extracts features from each example of the search target for subsequent matching with the features to be
        extracted from crops around fine glimpses.
        :param search_targets_original: [B, N_exmpls, 4, H, W], example images of the search target object,
                each example provides a different viewpoint. The last channel dimension contains a segmentation mask
        """
        self.search_target_embeddings = general_utils.apply_model_on_bb_regions(
            self.visual_embedder,
            search_targets_original.flatten(0, 1)[:, :3],
            torchvision.ops.masks_to_boxes(search_targets_original.flatten(0, 1)[:, 3]),
            resize=[224, 224],
            pad_to_square=True
        ).unflatten(0, search_targets_original.shape[:2])

    def get_similarity_score(self, input_image, mask, roi_bounding_box):
        return self.embedding_based_similarity_score(input_image, mask, roi_bounding_box)

    def embedding_based_similarity_score(self, input_image, mask, roi_bounding_box):
        """defined by the highest cosine similarity with the search target examples"""
        if mask is not None:
            input_image = input_image * mask[:, None].float()

        try:
            candidate_embedding = general_utils.apply_model_on_bb_regions(
                self.visual_embedder,
                input_image,
                roi_bounding_box, resize=[224, 224], pad_to_square=True
            )
        except RuntimeError as e:
            print("bounding box is invalid")
            return torch.tensor(-1.).to(input_image.device)

        if candidate_embedding.dim() > 2:
            sim = F.cosine_similarity(
                candidate_embedding[:, None], self.search_target_embeddings, -3
            ).mean([-1, -2]).max(-1).values
        else:
            sim = F.cosine_similarity(candidate_embedding[:, None], self.search_target_embeddings, -1).max(-1).values

        sim_score = sim.max()
        return sim_score


class STTDownstreamArch(DownstreamArch):
    def __init__(
            self,
            visual_embedder,
            stt_checkpoint,
            seg_mask_threshold=0.5,
            mask_cleaning_kernel_size=33,
            use_all_fine_glimpse_locs=False,
            crop_size=None,
            early_stopping_score_threshold=None,
            device=None
    ):
        super().__init__()
        self.visual_embedder = visual_embedder
        self.early_stopping_score_threshold = early_stopping_score_threshold

        self.crop_size = crop_size
        self.use_all_fine_glimpse_locs = use_all_fine_glimpse_locs
        self.mask_cleaning_kernel_size = mask_cleaning_kernel_size
        self.seg_mask_threshold = seg_mask_threshold

        from segment_this_thing import Foveator, build_segment_this_thing_b, SegmentThisThingPredictor
        from segment_this_thing import utils as stt_utils
        self.get_crop_bounds = stt_utils.get_crop_bounds
        self.get_centered_crop = stt_utils.get_centered_crop

        self.stt_input_size = (1280, 1280)

        foveation_pattern = Foveator(
            token_size=16, strides=[1, 2, 4, 6, 8], grid_sizes=[4, 4, 6, 8, 10]
        )

        model = build_segment_this_thing_b(
            num_tokens=foveation_pattern.get_num_tokens(),
            token_size=16
        )

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(
            stt_checkpoint,
            weights_only=True, map_location=device))
        model.to(device)

        self.predictor = SegmentThisThingPredictor(model, foveation_pattern)
        self.foveation_pattern = foveation_pattern

    def run_backbone(self, scene, location):
        """
        :param scene: [B, 3, H, W]
        :param location: [2]
        :return:
        """
        location = general_utils.from_normalized_to_pixel_based(location[None], scene.shape[-2:])[0]

        if self.crop_size is not None:
            crop_bounds = self.get_crop_bounds(
                location, self.crop_size[0]
            ).to(location.device)
            stt_input = self.get_centered_crop((scene * 255).byte().permute(1, 2, 0), crop_bounds)

            stt_input = VF.resize(stt_input.permute(2, 0, 1), list(self.stt_input_size)).permute(1, 2, 0)
            stt_location = torch.tensor([*self.stt_input_size], device=location.device) // 2
            crop_radius = torch.tensor([*self.crop_size], device=location.device) // 2
        else:
            stt_input = (scene * 255).byte().permute(1, 2, 0)
            stt_location = location
            crop_radius = torch.tensor([*self.stt_input_size], device=location.device) // 2

        left_top = location - crop_radius
        left_top[0] = left_top[0].clip(0, scene.shape[-1])
        left_top[1] = left_top[1].clip(0, scene.shape[-2])

        # this image will be used to cut out inputs for matching with visual_embedder
        input_image = VF.crop(scene, left_top[1], left_top[0], *[r*2 for r in crop_radius])

        masks, ious, foveal_image = self.predictor.get_prediction(stt_input, stt_location, return_foveation=True)
        seg_masks = []
        for m in masks:
            m = self.foveation_pattern.generate_foveated_visualization(m.unsqueeze(1)).sigmoid()
            if self.crop_size is not None:
                m = F.interpolate(m[None], self.crop_size)[0]
            seg_masks.append(m)
        seg_masks = torch.cat(seg_masks)
        seg_masks = (seg_masks > self.seg_mask_threshold).float().to(scene.device)

        n_masks = seg_masks.shape[0]
        top_sim = torch.tensor(-1)
        best_bounding_box = torch.tensor([[0., 0., 1., 1.]]).to(scene.device)
        all_bbs_and_scores = []

        for i_m in range(n_masks):
            mask = seg_masks[i_m: i_m+1]
            if mask.sum() == 0:
                mask = torch.ones_like(mask)

            bounding_box = torchvision.ops.masks_to_boxes(mask)
            cleaned_mask = mask
            if self.mask_cleaning_kernel_size is not None:
                try:
                    bounding_box, cleaned_mask = general_utils.noisy_masks_to_bboxes(
                        mask, kernel_size=self.mask_cleaning_kernel_size)
                except RuntimeError:
                    print("Masks could not be cleaned via erosion/dilation: probably they are too small")

            bounding_box, cleaned_mask = bounding_box.to(scene.device), cleaned_mask.to(scene.device)
            sim = self.get_similarity_score(input_image, cleaned_mask, bounding_box)

            # move the bounding box to the scene-based frame
            bounding_box += left_top.repeat(2)

            if sim > top_sim:
                top_sim = sim
                best_bounding_box = bounding_box

            all_bbs_and_scores.append((bounding_box.cpu(), sim.cpu()))

        return top_sim, best_bounding_box, all_bbs_and_scores

    def forward(self, scene, glimpse_locs, **kwargs):
        assert len(scene.shape) == 4 and scene.shape[0] == 1, "Only batch size of 1 is currently supported"

        if not self.use_all_fine_glimpse_locs:
            glimpse_locs = glimpse_locs[:, -1:]     # take only the last glimpse loc

        n_locs = glimpse_locs.shape[1]

        for i in range(n_locs):
            new_top_sim, new_best_bounding_box, _ = self.run_backbone(scene[0], glimpse_locs[0, i])
            if i == 0:
                top_sim, best_bounding_box = new_top_sim, new_best_bounding_box
            else:
                if new_top_sim > top_sim:
                    best_bounding_box = new_best_bounding_box
                    top_sim = new_top_sim

        pred_obj_center_loc = ((best_bounding_box[:, 2:] - best_bounding_box[:, :2]) // 2) + best_bounding_box[:, :2]

        if self.early_stopping_score_threshold is None:
            success = False
        else:
            success = top_sim >= self.early_stopping_score_threshold

        return dict(
            score=top_sim, success=success,
            bounding_boxes=best_bounding_box,
            pred_obj_center_loc=pred_obj_center_loc,
        )


class NidsNetDownstreamArch(DownstreamArch):
    """
    This is a compact implementation of NIDS-Net for seamless integration with CF-GAP.
    The implementation is based on the original repository https://github.com/IRVLUTD/NIDS-Net
    """

    class NoBBDetectedError(Exception):
        """Raised when GDINO/SAM produce no usable bounding boxes for a scene."""
        def __init__(self, m):
            super().__init__(m)

    # ------------------------------------------------------------------ #
    # Inlined feature adapters (from adapter.py)
    # ------------------------------------------------------------------ #
    class ModifiedClipAdapter(nn.Module):
        """Modified CLIP adapter (adds dropout + residual mixing)."""
        def __init__(self, c_in, reduction=4, ratio=0.6):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_in // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(c_in // reduction, c_in, bias=False),
                nn.ReLU(inplace=True),
            )
            self.ratio = ratio

        def forward(self, inputs):
            inputs = F.normalize(inputs, dim=-1, p=2)
            x = self.fc(inputs)
            x = self.ratio * x + (1 - self.ratio) * inputs
            return x

    class WeightAdapter(nn.Module):
        """Predicts a multiplicative (sigmoid-gated) weight per feature channel."""
        def __init__(self, c_in, reduction=4, scalar=10.0):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(c_in, c_in // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(c_in // reduction, c_in, bias=False),
                nn.ReLU(inplace=True),
            )
            self.scalar = scalar

        def forward(self, inputs):
            inputs = self.scalar * inputs
            x = self.fc(inputs)
            x = x.sigmoid()
            x = x * inputs
            return x

    # ------------------------------------------------------------------ #
    # Inlined utility functions
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_similarity(obj_feats, roi_feats):
        """Cosine similarity between object features and proposal features."""
        roi_feats = roi_feats.unsqueeze(-2)
        return torch.nn.functional.cosine_similarity(roi_feats, obj_feats, dim=-1)

    @staticmethod
    def stableMatching(preferenceMat):
        """Gale-Shapley stable matching over a (padded, square) preference matrix."""
        mDict = dict()
        engageMatrix = np.zeros_like(preferenceMat)
        for i in range(preferenceMat.shape[0]):
            tmp = preferenceMat[i]
            sortIndices = np.argsort(tmp)[::-1]
            mDict[i] = sortIndices.tolist()

        freeManList = list(range(preferenceMat.shape[0]))

        while freeManList:
            curMan = freeManList.pop(0)
            curWoman = mDict[curMan].pop(0)
            if engageMatrix[:, curWoman].sum() == 0:
                engageMatrix[curMan, curWoman] = 1
            else:
                engagedMan = np.where(engageMatrix[:, curWoman] == 1)[0][0]
                if preferenceMat[engagedMan, curWoman] > preferenceMat[curMan, curWoman]:
                    freeManList.append(curMan)
                else:
                    engageMatrix[engagedMan, curWoman] = 0
                    engageMatrix[curMan, curWoman] = 1
                    freeManList.append(engagedMan)
        return engageMatrix

    @staticmethod
    def mask_to_bbox(mask):
        rows, cols = torch.where(mask)
        if rows.size(0) == 0:
            return None
        x1 = torch.min(cols).item()
        y1 = torch.min(rows).item()
        x2 = torch.max(cols).item()
        y2 = torch.max(rows).item()
        return [x1, y1, x2, y2]

    @staticmethod
    def masks_to_bboxes(masks):
        assert masks.dim() == 3, "Input must be a 3D tensor."
        assert masks.dtype == torch.bool, "Input must be a boolean tensor."
        bboxes = []
        for i in range(masks.size(0)):
            bboxes.append(NidsNetDownstreamArch.mask_to_bbox(masks[i]))
        return bboxes

    @staticmethod
    def from_normalized_to_pixel_based(location, image_shape):
        """location: [..., 2] normalized in [-1, 1] -> integer pixel (x, y)."""
        return (torch.tensor(image_shape[-2:], device=location.device).flip(0)
                * (location + 1) / 2).int()

    @staticmethod
    def filter_boxes_containing_points(boxes, crop_top_left_loc, points, logits=None):
        """
        Keep only boxes (in crop-local xyxy) that contain at least one of ``points``
        (given in original-image space; ``crop_top_left_loc`` is the crop offset).
        """
        if boxes.size(0) == 0 or points.size(0) == 0:
            empty = torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device)
            if logits is not None:
                return empty, logits[:0]
            return empty

        crop_x, crop_y = crop_top_left_loc
        shift_tensor = torch.tensor([crop_x, crop_y], dtype=points.dtype, device=points.device)
        local_points = points - shift_tensor

        x1 = boxes[:, 0].unsqueeze(1)
        y1 = boxes[:, 1].unsqueeze(1)
        x2 = boxes[:, 2].unsqueeze(1)
        y2 = boxes[:, 3].unsqueeze(1)

        px = local_points[:, 0].unsqueeze(0)
        py = local_points[:, 1].unsqueeze(0)

        in_x = (px >= x1) & (px <= x2)
        in_y = (py >= y1) & (py <= y2)
        is_inside = in_x & in_y
        has_point = is_inside.any(dim=1)

        if logits is not None:
            return boxes[has_point], logits[has_point]
        return boxes[has_point]

    @staticmethod
    def get_foreground_mask_torch(masks, mask_size=24):
        """Resize masks (list of [1,H,W] or tensor) to (mask_size, mask_size) binary masks."""
        trafo = torchvision.transforms.Resize((mask_size, mask_size), interpolation=Image.BILINEAR)
        if isinstance(masks, list):
            resized_mask = []
            for m in masks:
                try:
                    resized_mask.append(trafo(m[None].float()))
                except RuntimeError:
                    resized_mask.append(torch.ones(1, mask_size, mask_size, device=m.device))
            resized_mask = torch.stack(resized_mask)
        else:
            resized_mask = trafo(masks)
        resized_mask = torch.where(resized_mask > 0.5, torch.ones_like(resized_mask), resized_mask).long()
        resized_mask = torch.where(
            resized_mask.sum([-1, -2], keepdim=True).expand(-1, -1, *resized_mask.shape[-2:]) == 0,
            torch.ones_like(resized_mask), resized_mask)
        return resized_mask

    @staticmethod
    def get_features_torch_based(images, masks, encoder, device="cuda", img_size=336):
        """Masked, average-pooled DINOv2 patch features. Returns [N, C]."""
        trafo = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if isinstance(images, list):
            preprocessed_imgs = []
            for i in range(len(images)):
                try:
                    _img = trafo(images[i])
                except RuntimeError:
                    _img = torch.zeros(3, img_size, img_size, device=device)
                preprocessed_imgs.append(_img)
            preprocessed_imgs = torch.stack(preprocessed_imgs)
            n_images = len(images)
        else:
            preprocessed_imgs = trafo(images)
            n_images = images.shape[0]

        mask_size = img_size // 14
        masks = NidsNetDownstreamArch.get_foreground_mask_torch(masks, mask_size).to(device)

        emb = encoder.forward_features(preprocessed_imgs.to(device))

        grid = emb["x_norm_patchtokens"].view(n_images, mask_size, mask_size, -1)
        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)
        return avg_feature

    @staticmethod
    def get_object_proposal_torch(raw_image, bboxs, masks, ratio=1.0):
        """
        Crop object proposals from a [C, H, W] image given xyxy ``bboxs`` and ``masks``.
        Returns ([], sel_rois, cropped_imgs, cropped_masks).
        """
        if not isinstance(raw_image, torch.Tensor):
            raw_image = torch.tensor(raw_image)
        if not isinstance(bboxs, torch.Tensor):
            bboxs = torch.tensor(bboxs)
        if not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks)

        C, image_height, image_width = raw_image.shape

        sel_rois = []
        cropped_masks = []
        cropped_imgs = []

        for ind in range(len(masks)):
            x0 = int(bboxs[ind][0].item())
            y0 = int(bboxs[ind][1].item())
            x1 = int(bboxs[ind][2].item())
            y1 = int(bboxs[ind][3].item())

            mask = masks[ind]
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)

            cropped_imgs.append(raw_image[:, y0:y1, x0:x1])
            cropped_masks.append(mask[y0:y1, x0:x1])

            sel_roi = dict()
            sel_roi['roi_id'] = int(ind)
            sel_roi['bbox'] = [int(x0 * ratio), int(y0 * ratio),
                               int((x1 - x0) * ratio), int((y1 - y0) * ratio)]
            sel_roi['roi_dir'] = None
            sel_roi['image_dir'] = None
            sel_roi['image_width'] = image_width
            sel_roi['image_height'] = image_height
            sel_roi['scale'] = int(1 / ratio)
            sel_rois.append(sel_roi)

        return [], sel_rois, cropped_imgs, cropped_masks

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        imsize: int = 448,
        use_adapter: bool = False,
        adapter_type: str = "weight",
        adapter_path=None,
        device=None,
        proposal_ratio: float = 0.25,
        do_stable_matching: bool = True,
        score_threshold: float = 0.6,
        use_sam_boxes: bool = False,
        min_crop_radius_to_scene_ratio=None,
        output_format: str = 'dict',  # 'dict' (DownstreamArch contract) or 'tuple'
        use_mean_loc: bool = False,
        use_loc_filter: bool = True,
        # --- backbone construction ---
        gdino_config_path: str = None,
        gdino_checkpoint_path: str = None,
        gdino_box_threshold: float = 0.15,
        gdino_text_threshold: float = 0.25,
        gdino_resize_input=(800, 1066),
        sam_vit_model: str = "vit_t",
        sam_checkpoint_path: str = None,
        encoder_name: str = "dinov2_vitl14_reg",
        encoder_feature_dim: int = 1024,
    ):
        super().__init__()
        self.use_loc_filter = use_loc_filter
        self.use_mean_loc = use_mean_loc
        self.output_format = output_format
        self.min_crop_radius_to_scene_ratio = min_crop_radius_to_scene_ratio
        self.use_sam_boxes = use_sam_boxes
        self.imsize = imsize
        self.use_adapter = use_adapter
        self.adapter_type = adapter_type
        self.proposal_ratio = proposal_ratio
        self.do_stable_matching = do_stable_matching
        self.score_threshold = score_threshold

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # --- DINOv2 encoder (feature extractor) ---
        print("Loading DINOv2 encoder...")
        self.encoder = torch.hub.load('facebookresearch/dinov2', encoder_name)
        self.encoder.to(self.device)
        self.encoder.eval()

        # --- optional feature adapter ---
        self.adapter = None
        if use_adapter:
            print(f"Loading {adapter_type} adapter...")
            if adapter_type == "clip":
                self.adapter = self.ModifiedClipAdapter(encoder_feature_dim, reduction=4, ratio=0.6).to(self.device)
            elif adapter_type == "weight":
                self.adapter = self.WeightAdapter(encoder_feature_dim, reduction=4).to(self.device)
            else:
                raise ValueError(f"Unknown adapter_type: {adapter_type}")

            if adapter_path and os.path.exists(adapter_path):
                self.adapter.load_state_dict(torch.load(adapter_path, map_location=self.device))
                self.adapter.eval()
                print("Adapter weights loaded successfully.")
            else:
                print(f"Warning: Adapter path not found: {adapter_path}")

        # --- GroundingDINO (object proposal boxes) ---
        import groundingdino.util.inference as groundingdino_inference
        import groundingdino.datasets.transforms as gdino_T
        self._gdino_inference = groundingdino_inference
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold
        self.gdino_resize_input = gdino_resize_input
        self.gdino_model = groundingdino_inference.load_model(gdino_config_path, gdino_checkpoint_path)
        self._gdino_transform = gdino_T.Compose([
            gdino_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # --- Segment Anything (proposal masks) ---
        from mobile_sam import sam_model_registry, SamPredictor
        self.sam_model = sam_model_registry[sam_vit_model](checkpoint=sam_checkpoint_path)
        self.sam_model.to(device=self.device)
        self.sam_model.eval()
        self.sam_predictor = SamPredictor(self.sam_model)

        self.cached_target_features = None
        self.use_cached_target_features = False

    # ------------------------------------------------------------------ #
    # Backbone wrappers (inlined GDINODetector / SegmentAnythingPredictor)
    # ------------------------------------------------------------------ #
    def _gdino(self, scene):
        """scene: [C, H, W] float tensor in [0, 1]. Returns (boxes_xyxy_pixel | None, logits)."""
        scene_preprocessed = VF.resize(scene, list(self.gdino_resize_input))
        scene_preprocessed = self._gdino_transform(scene_preprocessed, None)[0]

        channel, height, width = scene.shape
        boxes, logits, phrases = self._gdino_inference.predict(
            model=self.gdino_model,
            image=scene_preprocessed,
            caption="Objects",
            box_threshold=self.gdino_box_threshold,
            text_threshold=self.gdino_text_threshold,
            device=scene.device,
        )

        if len(boxes) > 0:
            boxes_cxcywh = boxes * torch.Tensor([width, height, width, height])
            boxes_xyxy = box_convert(boxes=boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
            boxes_xyxy = boxes_xyxy.clip(0, max(width, height))
        else:
            boxes_xyxy = None
        return boxes_xyxy, logits

    def _sam_predict(self, image, prompt_bboxes):
        """image: [C, H, W] float tensor; prompt_bboxes: [N, 4] xyxy pixel. Returns (boxes, masks)."""
        input_boxes = prompt_bboxes
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(prompt_bboxes, image.shape[-2:])
        preprocessed_image = self.sam_predictor.transform.apply_image_torch(image[None])
        self.sam_predictor.set_torch_image((preprocessed_image * 255).byte(), image.shape[-2:])
        transformed_boxes = transformed_boxes.to(image.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return input_boxes, masks

    # ------------------------------------------------------------------ #
    # Feature extraction & matching
    # ------------------------------------------------------------------ #
    def extract_scene_features(self, scene_image, loc_hist=None, output_dir=None):
        """
        scene_image: [C, H, W] tensor.
        loc_hist: [1, T, 2] glimpse locations in [-1, 1] to crop the scene and filter boxes.
        Returns (scene_features [N, D], sel_rois, bboxes [N, 4], masks).
        """
        left_top = None
        if loc_hist is not None and loc_hist.shape[-1] == 2:
            assert loc_hist.dim() == 3 and loc_hist.shape[0] == 1, "batched inputs are not yet implemented"
            loc_hist = loc_hist[0]

            patch_radius = int(min(scene_image.shape[-2:]) * self.min_crop_radius_to_scene_ratio)
            loc_hist_pixel = self.from_normalized_to_pixel_based(loc_hist, scene_image.shape[-2:])
            if self.use_mean_loc:
                center_crop_loc = loc_hist_pixel.float().mean(0).int()
            else:
                center_crop_loc = loc_hist_pixel[-1]

            left_top = center_crop_loc - patch_radius
            left_top[0] = left_top[0].clip(0, scene_image.shape[-1])
            left_top[1] = left_top[1].clip(0, scene_image.shape[-2])

            scene_image_crop = VF.crop(scene_image, left_top[1], left_top[0], patch_radius * 2, patch_radius * 2)
            bboxes, logits = self._gdino(scene_image_crop)
            if bboxes is None:
                raise self.NoBBDetectedError("GDINO found no BBs")

            filtered_bboxes, filtered_logits = self.filter_boxes_containing_points(
                bboxes.cpu(), left_top.cpu(), loc_hist_pixel.cpu(), logits=logits)

            bboxes = filtered_bboxes
            scene_image = scene_image_crop
        else:
            bboxes, logits = self._gdino(scene_image)

        if bboxes is None or bboxes.shape[0] == 0:
            raise self.NoBBDetectedError("GDINO found no BBs")

        _, masks = self._sam_predict(scene_image, bboxes)
        masks = masks.squeeze(1)

        if self.use_sam_boxes:
            bboxes = self.masks_to_bboxes(masks)  # accurate boxes from masks
            try:
                bboxes = torch.tensor(bboxes)
            except (RuntimeError, TypeError):
                raise self.NoBBDetectedError("SAM found no BBs")

        _, sel_rois, cropped_imgs, cropped_masks = self.get_object_proposal_torch(
            scene_image,
            bboxes,
            masks,
            ratio=self.proposal_ratio,
        )

        scene_features = self.get_features_torch_based(
            cropped_imgs,
            cropped_masks,
            self.encoder,
            device=self.device,
            img_size=self.imsize,
        )

        if self.use_adapter and self.adapter is not None:
            scene_features = self.adapter(scene_features)

        scene_features = nn.functional.normalize(scene_features, dim=1, p=2)

        if left_top is not None:
            bboxes = bboxes + left_top.repeat(2)[None].to(bboxes)

        for i in range(len(sel_rois)):
            sel_rois[i]['bbox'] = bboxes[i]

        return scene_features, sel_rois, bboxes, masks

    def extract_target_features(self, target_images, target_masks=None):
        """target_images: [N, 3, H, W]; target_masks: [N, 1, H, W]. Returns [N, D] L2-normed."""
        target_features = self.get_features_torch_based(
            target_images, target_masks,
            self.encoder,
            device=self.device,
            img_size=self.imsize,
        )
        if self.use_adapter and self.adapter is not None:
            target_features = self.adapter(target_features)
        target_features = nn.functional.normalize(target_features, dim=1, p=2)
        return target_features

    def match_proposals_to_targets(self, scene_features, target_features, proposals, num_target_views=1):
        """Match scene proposals to target object(s); returns a list of result dicts."""
        num_targets = len(target_features) // num_target_views

        sim_mat = self.compute_similarity(target_features, scene_features)
        sim_mat = sim_mat.view(len(scene_features), num_targets, num_target_views)

        sims, _ = torch.max(sim_mat, dim=2)  # [N_proposals, N_targets]
        max_ins_sim, initial_result = torch.max(sims, dim=1)

        num_proposals = len(proposals)
        results = []

        if self.do_stable_matching:
            sel_obj_ids = [str(v) for v in list(np.arange(num_targets))]
            sel_roi_ids = [str(v) for v in list(np.arange(len(scene_features)))]

            max_len = max(len(sel_roi_ids), len(sel_obj_ids))
            sel_sims_symmetric = torch.ones((max_len, max_len)) * -1
            sel_sims_symmetric[:len(sel_roi_ids), :len(sel_obj_ids)] = sims.clone()

            pad_len = abs(len(sel_roi_ids) - len(sel_obj_ids))
            if len(sel_roi_ids) > len(sel_obj_ids):
                pad_obj_ids = [str(i) for i in range(num_targets, num_targets + pad_len)]
                sel_obj_ids += pad_obj_ids
            elif len(sel_roi_ids) < len(sel_obj_ids):
                pad_roi_ids = [str(i) for i in range(len(sel_roi_ids), len(sel_roi_ids) + pad_len)]
                sel_roi_ids += pad_roi_ids

            matchedMat = self.stableMatching(sel_sims_symmetric.detach().cpu().numpy())
            Matches = dict()
            for i in range(matchedMat.shape[0]):
                tmp = matchedMat[i, :]
                a = tmp.argmax()
                Matches[sel_roi_ids[i]] = sel_obj_ids[int(a)]

            for k, v in Matches.items():
                if int(k) >= num_proposals:
                    break
                result = dict()
                result['proposal_id'] = int(k)
                result['target_id'] = int(v)
                result['bbox'] = proposals[int(k)]['bbox']
                result['score'] = float(sims[int(k), int(v)])
                result['image_width'] = proposals[int(k)]['image_width']
                result['image_height'] = proposals[int(k)]['image_height']
                results.append(result)
        else:
            for i in range(num_proposals):
                if float(max_ins_sim[i]) < self.score_threshold:
                    continue
                result = dict()
                result['proposal_id'] = i
                result['target_id'] = initial_result[i].item()
                result['bbox'] = proposals[i]['bbox']
                result['score'] = float(max_ins_sim[i])
                result['image_width'] = proposals[i]['image_width']
                result['image_height'] = proposals[i]['image_height']
                results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # DownstreamArch interface
    # ------------------------------------------------------------------ #
    def reset(self):
        self.cached_target_features = None
        self.use_cached_target_features = False

    def initialize(self, search_targets_original):
        """search_targets_original: [1, N, 4, H, W] (RGB + segmentation mask)."""
        assert search_targets_original.shape[0] == 1
        target_objects = search_targets_original[0, :, :3]
        target_masks = search_targets_original[0, :, 3:]
        self.cached_target_features = self.extract_target_features(target_objects, target_masks)
        self.use_cached_target_features = True

    def _empty_result(self):
        if self.output_format == 'dict':
            return dict(
                score=torch.tensor(-1.),
                bounding_boxes=torch.tensor([[0., 0., 1., 1.]]),
                success=False,
                pred_obj_center_loc=torch.ones(1, 2),
            )
        else:
            return torch.ones(1, 2), torch.tensor([[0., 0., 1., 1.]]), dict(
                all_scores=torch.zeros(1, 1), score=torch.tensor(0.))

    @torch.no_grad()
    def forward(self, scene, glimpse_locs, **kwargs):
        """
        scene: [1, 3, H, W]; glimpse_locs: [1, T, 2] normalized in [-1, 1].
        Returns the DownstreamArch result dict (or a tuple if output_format == 'tuple').
        """
        assert scene.shape[0] == 1, "batched inference is not supported yet"
        loc_hist = glimpse_locs if self.use_loc_filter else None
        scene = scene[0]

        assert self.cached_target_features is not None, "Call initialize(...) before forward(...)."

        try:
            scene_features, proposals, bboxes, masks = self.extract_scene_features(scene, loc_hist=loc_hist)
        except self.NoBBDetectedError as e:
            print(e)
            return self._empty_result()

        target_features = self.cached_target_features
        results = self.match_proposals_to_targets(
            scene_features,
            target_features,
            proposals,
            num_target_views=target_features.shape[0],
        )

        if len(results) == 0:
            return self._empty_result()

        best_idx = np.array([r['score'] for r in results]).argmax()
        best_score = torch.tensor(results[best_idx]['score'])

        final_bbox = results[best_idx]['bbox'][None]
        pred_location = ((final_bbox[:, 2:] - final_bbox[:, :2]) // 2) + final_bbox[:, :2]

        if self.output_format == 'tuple':
            return pred_location, final_bbox, dict(
                all_scores=torch.tensor([r['score'] for r in results])[None],
                all_bounding_boxes=torch.stack([r['bbox'] for r in results]),
                score=best_score[None])
        else:
            return dict(
                score=best_score, bounding_boxes=final_bbox, success=False,
                pred_obj_center_loc=pred_location,
                all_bounding_boxes=final_bbox[:, None],
                all_scores=best_score,
            )


class GDINOObjectDetectionDownstreamArch(DownstreamArch):
    def __init__(
            self,
            crop_to_scene_size_ratio,
            visual_embedder,
            gdino_checkpoint_path,
            gdino_config_path,
            gdino_input_size=(800, 1066),
            use_loc_to_select_bbs=False,
            use_any_fine_loc_to_select_bbs=False,  # if False, only the last fine glimpse location is used for selection
    ):
        super().__init__()

        self.use_any_fine_loc_to_select_bbs = use_any_fine_loc_to_select_bbs
        self.use_loc_to_select_bbs = use_loc_to_select_bbs or use_any_fine_loc_to_select_bbs

        self.visual_embedder = visual_embedder
        self.gdino_input_size = gdino_input_size
        self.crop_to_scene_size_ratio = crop_to_scene_size_ratio

        import groundingdino.util.inference as groundingdino_inference
        self.groundingdino_inference = groundingdino_inference

        self.model_groundingdino = groundingdino_inference.load_model(gdino_config_path, gdino_checkpoint_path)

        import groundingdino.datasets.transforms as T
        self.preprocess_transform = T.Compose(
            [
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def is_within_bb(pred_location, gt_scene_bounding_box):
        within_bb_x = (pred_location[:, 0] >= gt_scene_bounding_box[:, 0]) * (
                    pred_location[:, 0] <= gt_scene_bounding_box[:, 2])
        within_bb_y = (pred_location[:, 1] >= gt_scene_bounding_box[:, 1]) * (
                    pred_location[:, 1] <= gt_scene_bounding_box[:, 3])
        return within_bb_x * within_bb_y

    def run_backbone(self, scene):
        scene_preprocessed = VF.resize(scene, self.gdino_input_size)
        scene_preprocessed = self.preprocess_transform(scene_preprocessed, None)[0]

        channel, height, width = scene.shape
        boxes, logits, phrases = self.groundingdino_inference.predict(
            model=self.model_groundingdino,
            image=scene_preprocessed,
            caption="Objects",
            box_threshold=0.15,
            text_threshold=0.25,
            device=scene.device,
        )

        if len(boxes) > 0:
            boxes_cxcywh = boxes * torch.Tensor([width, height, width, height])
            boxes_xywh = box_convert(boxes=boxes_cxcywh, in_fmt="cxcywh", out_fmt="xywh")
        else:
            boxes_xywh = []

        return boxes_xywh

    def forward(self, scene, glimpse_locs, **kwargs):
        assert len(scene.shape) == 4 and scene.shape[0] == 1, "Only batch size of 1 is currently supported"

        crop_size = int(min(scene.shape[-2:]) * self.crop_to_scene_size_ratio)
        # take the last fine glimpse location as a center of the crop to be passed to the downstream architecture
        center_crop_loc = general_utils.from_normalized_to_pixel_based(glimpse_locs, scene.shape[-2:])[0, -1]

        left_top = center_crop_loc - crop_size
        left_top[0] = left_top[0].clip(0, scene.shape[-1])
        left_top[1] = left_top[1].clip(0, scene.shape[-2])

        scene_crop = VF.crop(scene, left_top[1], left_top[0], crop_size * 2, crop_size * 2)

        boxes_xywh = self.run_backbone(scene_crop[0])

        if len(boxes_xywh) > 0:
            boxes_xywh = boxes_xywh.int()

            boxes_xywh_scene = boxes_xywh.clone()
            boxes_xywh_scene[:, :2] = boxes_xywh[:, :2] + left_top.cpu()

            all_object_crops = []
            selected_boxes_xywh_scene = []
            for i, box in enumerate(boxes_xywh_scene):
                try:
                    if self.use_loc_to_select_bbs:
                        if self.use_any_fine_loc_to_select_bbs:
                            check_locs = general_utils.from_normalized_to_pixel_based(glimpse_locs, scene.shape[-2:])[0]
                        else:
                            check_locs = center_crop_loc[None]

                        within_bb = self.is_within_bb(
                            check_locs.cpu(),
                            box_convert(box[None].cpu(), in_fmt="xywh", out_fmt='xyxy').expand(check_locs.shape[0], -1)
                        ).any()
                        if not within_bb:
                            continue
                    object_crop = VF.crop(scene, box[1], box[0], box[3], box[2])
                    object_crop = general_utils.pad_image_to_square(object_crop)
                    object_crop = VF.resize(object_crop, [224, 224])    # resize to expected dims of DINO (matching)

                    all_object_crops.append(object_crop)
                    selected_boxes_xywh_scene.append(box)
                except RuntimeError:
                    continue
            if len(all_object_crops) > 0:
                all_object_crops = torch.cat(all_object_crops)
                boxes_xywh_scene = torch.stack(selected_boxes_xywh_scene)

                all_object_crops = self.visual_embedder(all_object_crops)
                sim = F.cosine_similarity(all_object_crops[:, None], self.search_target_embeddings, -1).max(-1).values
                top_sim_score, best_bb_idx = sim.max(), sim.argmax()

                best_bounding_box = boxes_xywh_scene[best_bb_idx]
                best_bounding_box = box_convert(best_bounding_box[None], in_fmt="xywh", out_fmt='xyxy')

        if len(boxes_xywh) == 0 or len(all_object_crops) == 0:
            # no bounding box detected
            top_sim_score = -1.
            best_bounding_box = torch.tensor([[0., 0., 1., 1.]], device=scene.device)

        pred_obj_center_loc = ((best_bounding_box[:, 2:] - best_bounding_box[:, :2]) // 2) + best_bounding_box[:, :2]
        return dict(
            score=top_sim_score,
            bounding_boxes=best_bounding_box,
            pred_obj_center_loc=pred_obj_center_loc,
            success=False,
        )


class SAMDownstreamArch(DownstreamArch):
    def __init__(
            self,
            crop_to_scene_size_ratio,
            n_prompt_locs,
            visual_embedder,
            checkpoint_path,
            early_stopping_threshold=torch.inf,
            use_big_sam=False,
    ):
        super().__init__()

        if not use_big_sam:
            from mobile_sam import sam_model_registry, SamPredictor
            model_type = "vit_t"
        else:
            from segment_anything import sam_model_registry, SamPredictor
            model_type = "vit_l"

        self.sam_backbone = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_backbone.eval()

        self.sam_image_size = [1024, 1024]
        self.predictor = SamPredictor(self.sam_backbone)

        if use_big_sam:
            from segment_anything import SamAutomaticMaskGenerator
        else:
            from mobile_sam import SamAutomaticMaskGenerator
        self.automatic_mask_generator_cls = SamAutomaticMaskGenerator

        self.visual_embedder = visual_embedder
        self.n_prompt_locs = n_prompt_locs
        self.early_stopping_threshold = early_stopping_threshold
        self.crop_to_scene_size_ratio = crop_to_scene_size_ratio

    def forward(self, scene, glimpse_locs, **kwargs):
        assert len(scene.shape) == 4 and scene.shape[0] == 1, "Only batch size of 1 is currently supported"

        crop_size = int(min(scene.shape[-2:]) * self.crop_to_scene_size_ratio)
        # take the last fine glimpse location as a center of the crop to be passed to the downstream architecture
        center_crop_loc = general_utils.from_normalized_to_pixel_based(glimpse_locs, scene.shape[-2:])[0, -1]

        left_top = center_crop_loc - crop_size
        left_top[0] = left_top[0].clip(0, scene.shape[-1])
        left_top[1] = left_top[1].clip(0, scene.shape[-2])

        scene_crop = VF.crop(scene, left_top[1], left_top[0], crop_size * 2, crop_size * 2)

        # use glimpse locs as spatial prompts for SAM
        # note: the current invoking of SAM is very inefficient since it uses 'AutomaticMaskGenerator' interface
        #       that requires moving inputs from CUDA to CPU and back to CUDA. It is done so to use various utils
        #       of 'AutomaticMaskGenerator' for cleaning masks, non-maximum-suppression etc.
        prompted_locs = general_utils.from_normalized_to_pixel_based(
            glimpse_locs[:, -self.n_prompt_locs:], scene.shape[-2:])[0]
        prompted_locs = prompted_locs - left_top
        mask = (prompted_locs >= 0) & (prompted_locs < crop_size * 2)
        mask = mask.all(dim=1)       # A location is valid only if both its x and y coordinates are in range
        prompted_locs = prompted_locs[mask]
        prompted_locs = general_utils.from_normalized_to_pixel_based(
            general_utils.from_pixel_based_to_normalized(prompted_locs, scene_crop.shape[-2:]),
            self.sam_image_size
        )
        scene_crop = VF.resize(general_utils.auto_adjust_contrast(scene_crop), self.sam_image_size)
        self.predictor.set_torch_image((scene_crop * 255).int(), scene_crop.shape[-2:])
        prompted_points = (prompted_locs / 1024).split(1, 0)
        mask_generator = self.automatic_mask_generator_cls(
            self.sam_backbone, points_per_side=None, point_grids=[p.numpy() for p in prompted_points]
        )
        mask_data = mask_generator.generate(scene_crop[0].permute(1, 2, 0).numpy())
        n_masks = len(mask_data)

        top_sim = torch.tensor(-1)
        best_bounding_box = torch.tensor([[0., 0., 1., 1.]]).to(scene.device)
        for i_m in range(n_masks):
            bounding_box = torch.tensor(mask_data[i_m]['bbox'])[None].float()
            bounding_box[:, 2:] += bounding_box[:, :2]
            cleaned_mask = torch.tensor(mask_data[i_m]['segmentation']).float()[None]

            bounding_box, cleaned_mask = bounding_box.to(scene.device), cleaned_mask.to(scene.device)
            try:
                sim = self.get_similarity_score(scene_crop, cleaned_mask, bounding_box)
            except RuntimeError:
                # invalid bounding box
                sim = torch.tensor(-1.).to(scene.device)
                bounding_box = torch.tensor([[0., 0., 1., 1.]]).to(scene.device)

            # move the bounding box to the scene-based frame
            bounding_box *= (crop_size * 2) / self.sam_image_size[0]
            bounding_box += left_top.repeat(2)

            if sim > top_sim:
                top_sim = sim
                best_bounding_box = bounding_box

        if top_sim > self.early_stopping_threshold:
            success = True
        else:
            success = False

        pred_obj_center_loc = ((best_bounding_box[:, 2:] - best_bounding_box[:, :2]) // 2) + best_bounding_box[:, :2]

        return dict(
            score=top_sim,
            success=success,
            bounding_boxes=best_bounding_box,
            pred_obj_center_loc=pred_obj_center_loc,
        )
