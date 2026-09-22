import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision


def compute_bounding_box_metrics(pred_bbs, target_bbs):
    """
    :param pred_bbs: shape [N, 4]
    :param target_bbs: shape [N, 4]
    :return:
    """
    preds = [
        {
            # The boxes keyword should contain an [N,4] tensor,
            # where N is the number of detected boxes with boxes of the format
            # [xmin, ymin, xmax, ymax] in absolute image coordinates
            "boxes": pred_bbs,
            # The scores keyword should contain an [N,] tensor where
            # each element is confidence score between 0 and 1
            "scores": torch.ones(pred_bbs.shape[0]),
            # The labels keyword should contain an [N,] tensor
            # with integers of the predicted classes
            "labels": torch.zeros(pred_bbs.shape[0], dtype=torch.int),
            # The masks keyword should contain an [N,H,W] tensor,
            # where H and W are the image height and width, respectively,
            # with boolean masks. This is only required when iou_type is `segm`.
            # "masks": BoolTensor([mask_pred]),
        }
    ]
    target = [
        {
            "boxes": target_bbs,
            "labels": torch.zeros(pred_bbs.shape[0], dtype=torch.int),
        }
    ]
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, target)
    all_metrics = metric.compute()
    result = {}
    for metric_name, value in all_metrics.items():
        if 'map' in metric_name:
            result[metric_name] = value
    return result


def evaluate(model, dataset, device='cpu'):
    dataset = DataLoader(dataset, batch_size=1)
    accumulated_bb_metrics = None
    for i, sample in enumerate(dataset):
        print(f"Running sample {i + 1}")
        output = model(sample['scene'].to(device), sample['target_object_images'].to(device))
        pred_bounding_box = output[1]
        bb_metrics = compute_bounding_box_metrics(pred_bounding_box.cpu(), sample['gt_scene_bounding_box'].cpu())
        if accumulated_bb_metrics is None:
            accumulated_bb_metrics = bb_metrics
        else:
            for metric_name, value in bb_metrics.items():
                accumulated_bb_metrics[metric_name] += value

    print("Results:")
    for metric_name, value in accumulated_bb_metrics.items():
        print(f"Metric {metric_name}: {value / (i + 1)}")