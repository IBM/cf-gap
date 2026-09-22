import glob
import json
import os
from typing import Literal

import torch
import torchvision
import torchvision.transforms.functional as VF


from eval_datasets.pascal2coco import readXML


def load_image_nvjpegl_cpu(image_path):
    return torchvision.io.decode_image(image_path) / 255


def load_single_search_target_example(path, resize=None, pad_size=None, include_mask=False, with_bounding_box=False,
                                      mask_background=True, mask_value=0., image_dir_name='images', mask_dir_name='masks'):
    img = load_image_nvjpegl_cpu(path)
    try:
        if os.path.isfile(path.replace(image_dir_name, mask_dir_name)):
            mask = torchvision.io.decode_image(path.replace(image_dir_name, mask_dir_name), torchvision.io.ImageReadMode.GRAY) / 255
        elif os.path.isfile(path.replace(image_dir_name, mask_dir_name).replace('jpg', 'png')):
            mask = torchvision.io.decode_image(path.replace(image_dir_name, mask_dir_name).replace('jpg', 'png'), torchvision.io.ImageReadMode.GRAY) / 255
        else:
            raise RuntimeError
        if mask_background:
            if mask_value > 0.:
                img = img * (mask * mask_value)
            else:
                img = img * mask
    except RuntimeError:
        print(f"No mask exists for {path}")
        if include_mask:
            assert FileNotFoundError, f"No mask exists for {path}"

    if pad_size is not None:
        img = VF.pad(img, pad_size)
        if include_mask:
            mask = VF.pad(mask, pad_size)

    if resize is not None:
        img = VF.resize(img, resize)
        if include_mask:
            mask = VF.resize(mask, resize, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    if include_mask:
        img = torch.cat([img, mask], 0)

    if with_bounding_box:
        bounding_box = torchvision.ops.masks_to_boxes(mask)

    if with_bounding_box:
        return img, bounding_box
    else:
        return img

def load_search_target_all_examples(
        path, resize=None, pad_size=None, include_mask=False, with_bounding_box=False,
        every_n_example=None, mask_background=True, mask_value=0., target_idx=None,
        image_folder_name='images', mask_dir_name='masks', format='.jpg'
):
    all_examples, all_bounding_boxes = list(), list()
    file_paths = glob.glob(f'{path}/{image_folder_name}/*{format}')
    file_paths = sorted(file_paths, key=lambda _file_name: int(_file_name.split('/')[-1].split(format)[0].split('_')[-1]))
    if every_n_example is not None:
        file_paths = file_paths[0::every_n_example]
    if target_idx is not None:
        file_paths = file_paths[target_idx:target_idx+1]

    for p in file_paths:
        example = load_single_search_target_example(
            p, resize, pad_size, include_mask, with_bounding_box,
            mask_background=mask_background, mask_value=mask_value,
            image_dir_name=image_folder_name, mask_dir_name=mask_dir_name
        )

        if with_bounding_box:
            example, bounding_box = example
            all_bounding_boxes.append(bounding_box)

        all_examples.append(example)

    if with_bounding_box:
        return torch.stack(all_examples), torch.cat(all_bounding_boxes)
    else:
        return torch.stack(all_examples)


def resize_bounding_box(bounding_box, new_size, old_size):
    resize_ratio = torch.tensor(new_size) / torch.tensor(old_size)

    xmin, ymin, xmax, ymax = bounding_box
    resize_ratio_y, resize_ratio_x = resize_ratio[0], resize_ratio[1]
    bounding_box = (
        int(xmin * resize_ratio_x),
        int(ymin * resize_ratio_y),
        int(xmax * resize_ratio_x),
        int(ymax * resize_ratio_y),
    )
    return torch.tensor(bounding_box)


def recenter_image_around_bbox(
        images: torch.Tensor,
        bboxes: torch.Tensor,
        padding_value: int = 0.
):
    """
    Re-centers a batch of images around their bounding box centers, keeping the
    original image dimensions by adding padding. Also updates bbox coordinates.

    Args:
        images (torch.Tensor): A batch of images of shape (B, C, H, W).
        bboxes (torch.Tensor): A batch of bounding boxes of shape (B, 4)
                               in the format (xmin, ymin, xmax, ymax).
        padding_value (int): The value to use for padding (e.g., 0 for black).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - torch.Tensor: Batch of re-centered images (B, C, H, W).
            - torch.Tensor: Batch of updated bboxes (B, 4) relative to the
                            newly centered image frame.
    """
    batch_size, _, H, W = images.shape
    device = images.device

    # Initialize tensors to store the results
    recentered_images = torch.zeros_like(images)
    new_bboxes = torch.zeros_like(bboxes)

    for i in range(batch_size):
        image = images[i]
        bbox = bboxes[i]

        # 1. Calculate the required shift
        # --------------------------------
        # Bbox center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Image center
        img_center_x = W / 2
        img_center_y = H / 2

        # The translation needed to move the bbox center to the image center
        tx = img_center_x - center_x
        ty = img_center_y - center_y

        # 2. Apply the affine transformation
        # ----------------------------------
        # F.affine handles the shifting and padding in one step.
        recentered_image = VF.affine(
            img=image,
            angle=0,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=padding_value
        )
        recentered_images[i] = recentered_image

        # 3. Update bounding box coordinates
        # ----------------------------------
        # The bounding box is shifted by the same amount as the image.
        new_bboxes[i] = bbox + torch.tensor([tx, ty, tx, ty], device=device)

    return recentered_images, new_bboxes


class HRInsDet(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir_scenes, root_dir_objects,
            n_samples=None,
            resize=None, resize_targets=None, every_n_example=None,
            n_triples=None,
            size_type: Literal['small', 'medium', 'large', None] = None,
            with_names=False,
    ):
        super().__init__()

        self.with_names = with_names

        self.resize_targets = resize_targets
        self.resize = resize
        self.root_dir_scenes = root_dir_scenes
        self.root_dir_objects = root_dir_objects
        self.every_n_example = every_n_example

        self.image_file_paths = glob.glob(f"{self.root_dir_scenes}/images/*.jpg")
        self.image_file_paths = sorted(
            self.image_file_paths, key=lambda _file_name: int(_file_name.split('/')[-1].split('.jpg')[0].split('_')[-1]))

        if n_samples is not None:
            self.image_file_paths = self.image_file_paths[:n_samples] if n_samples > 0 else self.image_file_paths[n_samples:]

        self.object_image_paths = glob.glob(f"{self.root_dir_objects}/*")
        self.object_names = [s.split('/')[-1] for s in self.object_image_paths]

        self.scene_object_bounding_box_triplets = list()
        for scene_image_path in self.image_file_paths:
            annotations = readXML(scene_image_path.replace('images', 'annotations').replace('.jpg', '.xml'))
            meta_info, bounding_boxes_info = annotations
            for single_bb in bounding_boxes_info:
                if single_bb['name'] in self.object_names:
                    new_data_point = (
                        scene_image_path,
                        os.path.join(root_dir_objects, single_bb['name']),
                        (single_bb['xmin'], single_bb['ymin'], single_bb['xmax'], single_bb['ymax'])
                    )
                    if size_type is not None:
                        bb_area = (single_bb['xmax'] - single_bb['xmin']) * (single_bb['ymax'] - single_bb['ymin'])
                        if size_type == 'small' and bb_area < 200**2:
                            self.scene_object_bounding_box_triplets.append(new_data_point)
                        if size_type == 'medium' and bb_area >= 200**2 and bb_area <= 400**2:
                            self.scene_object_bounding_box_triplets.append(new_data_point)
                        if size_type == 'large' and bb_area > 400**2:
                            self.scene_object_bounding_box_triplets.append(new_data_point)
                    else:
                        self.scene_object_bounding_box_triplets.append(new_data_point)
        if n_triples is not None:
            self.scene_object_bounding_box_triplets = self.scene_object_bounding_box_triplets[:n_triples]

    def __len__(self):
        return len(self.scene_object_bounding_box_triplets)

    def __getitem__(self, idx):
        scene_path, object_path, scene_gt_bounding_box = self.scene_object_bounding_box_triplets[idx]
        scene_gt_bounding_box = torch.tensor(scene_gt_bounding_box)

        scene = load_image_nvjpegl_cpu(scene_path)
        if self.resize is not None:
            scene_gt_bounding_box = resize_bounding_box(scene_gt_bounding_box, self.resize, scene.shape[-2:])
            scene = VF.resize(scene, self.resize)

        objects, object_bounding_boxes = load_search_target_all_examples(
            object_path, self.resize_targets, include_mask=True, with_bounding_box=True, every_n_example=self.every_n_example)

        return dict(
            scene=scene, target_object_images=objects,
            gt_scene_bounding_box=scene_gt_bounding_box,
            target_object_bounding_boxes=object_bounding_boxes,
            scene_path=scene_path, object_path=object_path,
        )


class Robotools(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir_scenes, root_dir_objects,
            n_samples=None,
            resize=None, resize_targets=None, every_n_example=None,
            target_crop_size=None,
            with_names=False,
            every_n_sample=None,
    ):
        super().__init__()

        from detectron2.data import DatasetCatalog
        from detectron2.data.datasets.coco import load_coco_json

        self.resize_targets = resize_targets
        self.resize = resize
        self.root_dir_scenes = root_dir_scenes
        self.root_dir_objects = root_dir_objects
        self.every_n_example = every_n_example
        self.with_names = with_names

        self.object_image_paths = [f.path for f in os.scandir(root_dir_objects) if f.is_dir()]
        self.object_image_paths = sorted(self.object_image_paths, key=lambda _file_name: int(_file_name.split('/')[-1].split('_')[-1]))

        DatasetCatalog.register(
            "robotools",
            lambda: load_coco_json(os.path.join(root_dir_scenes, 'scene_gt_coco_all.json'), root_dir_scenes))

        self.annotations = DatasetCatalog.get("robotools")
        self.target_crop_size = target_crop_size

        if every_n_sample is not None:
            self.annotations = self.annotations[0::every_n_sample]
        if n_samples is not None:
            self.annotations = self.annotations[:n_samples] if n_samples > 0 else self.annotations[n_samples:]

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def crop_and_update_bboxes(
            images: torch.Tensor,
            bboxes: torch.Tensor,
            crop_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Crops a batch of images around the center of their bounding boxes and
        updates the bounding box coordinates to be relative to the new cropped image.

        Args:
            images (torch.Tensor): A batch of images of shape (B, C, H, W).
            bboxes (torch.Tensor): A batch of bounding boxes of shape (B, 4)
                                   in the format (xmin, ymin, xmax, ymax).
            crop_size (int): The desired height and width of the square crop.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - torch.Tensor: Batch of cropped images (B, C, crop_size, crop_size).
                - torch.Tensor: Batch of updated bboxes (B, 4) relative to the crop.
        """
        batch_size, _, H, W = images.shape
        device = images.device

        # Initialize tensors to store the results
        cropped_images = torch.zeros((batch_size, images.shape[1], crop_size, crop_size), device=device)
        new_bboxes = torch.zeros((batch_size, 4), device=device)

        for i in range(batch_size):
            image = images[i]
            bbox = bboxes[i]

            # 1. CROP THE IMAGE (same logic as before)
            # ----------------------------------------
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Calculate the top-left corner for the crop
            top = int(center_y - crop_size / 2)
            left = int(center_x - crop_size / 2)

            # Clamp crop coordinates to be within the original image bounds
            clamped_top = max(0, min(top, H - crop_size))
            clamped_left = max(0, min(left, W - crop_size))

            # Perform the crop
            cropped_image = VF.crop(image, clamped_top, clamped_left, crop_size, crop_size)
            cropped_images[i] = cropped_image

            # 2. UPDATE THE BOUNDING BOX
            # --------------------------
            # Translate the original bbox coordinates by the crop's top-left corner
            translated_bbox = bbox - torch.tensor([clamped_left, clamped_top, clamped_left, clamped_top], device=device)

            # Clamp the new coordinates to the crop dimensions [0, crop_size]
            # This handles cases where the original bbox was partially outside the crop area
            updated_bbox = torch.clamp(translated_bbox, 0, crop_size)
            new_bboxes[i] = updated_bbox

        return cropped_images, new_bboxes

    def __getitem__(self, idx):
        data_dict_item = self.annotations[idx]
        annotation = data_dict_item['annotations'][0]

        scene_gt_bounding_box = torch.tensor(annotation['bbox'])
        scene_gt_bounding_box[2:] = scene_gt_bounding_box[:2] + scene_gt_bounding_box[2:]

        scene = load_image_nvjpegl_cpu(data_dict_item['file_name'])
        if self.resize is not None:
            scene, scene_gt_bounding_box = self.crop_and_update_bboxes(
                scene[None], scene_gt_bounding_box[None], self.resize)
            scene, scene_gt_bounding_box = scene[0], scene_gt_bounding_box[0]

        objects, object_bounding_boxes = load_search_target_all_examples(
            self.object_image_paths[annotation['category_id']-1],
            self.resize_targets, include_mask=True, with_bounding_box=True, every_n_example=self.every_n_example,
            image_folder_name='rgb', format='.png', mask_dir_name='mask',
        )

        objects, object_bounding_boxes = recenter_image_around_bbox(
            objects, object_bounding_boxes)

        return dict(
            scene=scene, target_object_images=objects,
            gt_scene_bounding_box=scene_gt_bounding_box, object_bounding_boxes=object_bounding_boxes,
            scene_path=data_dict_item['file_name'], object_path=self.object_image_paths[annotation['category_id']-1],
        )
