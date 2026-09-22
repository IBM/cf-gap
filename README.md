# Coarse-to-Fine Glimpse-based Active Perception (CF-GAP)

Official implementation of the ECCV 2026 paper
**[Task-driven Processing with Coarse-to-Fine Glimpse-based Active Perception](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/14294.pdf)**
by Oleh Kolner, Thomas Ortner, Stanisław Woźniak, and Angeliki Pantazi.

## Overview

CF-GAP is a **task-driven front-end** that wraps an existing instance detector and feeds it only
the task-relevant regions at full resolution. Given a scene and a few masked example views of
a search target, it:

1. **Builds a coarse priority map** over a downsampled scene to rank where the target is likely to
   be — `coarse_map.py` (`CoarseSearchMapGeneration`, MobileNetV3 backbone).
2. **Refines each coarse glimpse with fine, log-polar glimpses** that iteratively re-center on the
   target — `fine_glimpsing/` (`LogPolarSensor` + a learned `FineSearchMapGeneration`, weights in
   `fine_search_map_checkpoint.pt`).
3. **Runs a swappable downstream detector** at high resolution on the attended region —
   `downstream_architectures.py`.
4. **Applies inhibition-of-return** to suppress visited regions and move on —
   `IoRMasker` in `coarse_map.py`.

The full loop is orchestrated by `CoarseToFineGAP` in [coarse_to_fine_gap.py](coarse_to_fine_gap.py).
Acting purely as a front-end, CF-GAP improves Average Precision by **up to ~20%** across several
state-of-the-art instance detectors on the **HR-InsDet** and **Robotools** benchmarks, letting
lightweight detectors rival much larger ones.

## Setup

The code targets **Python 3.12**. Install the core requirements:

```
pip install -r requirements.txt
```

Install the downstream architectures you intend to use, following the instructions in their
original repositories (you only need the ones your chosen configs use):

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [Segment This Thing (STT)](https://github.com/facebookresearch/segment_this_thing)

[detectron2](https://github.com/facebookresearch/detectron2) is only required to prepare the
Robotools dataset; it is not needed otherwise. DINOv2, used for feature matching, is downloaded
automatically via `torch.hub`.

Datasets can be downloaded from their official repositories:

- [HR-InsDet](https://github.com/insdet/instance-detection)
- [Robotools](https://github.com/Jaraxxus-Me/VoxDet)

Finally, set the dataset and checkpoint paths in [project_definitions.py](project_definitions.py)
(`DATA_PATH_HR_INSDET`, `DATA_PATH_ROBOTOOLS`, and the per-detector checkpoints/configs). The fine
search-map checkpoint (`CHECKPOINT_PATH_FINE_SEARCH_MAP`) already points at the bundled
`fine_search_map_checkpoint.pt`.

## Running experiments

Each dataset × downstream-detector combination is provided as a standalone config script. Run one
by executing it with the project root on `PYTHONPATH`, e.g.:

```
PYTHONPATH=<path/to/this/project> python experiments/hr_insdet/cf_gap_grounding_dino.py
```

## Demo

The [demo notebook](experiments/demo.ipynb) lets you evaluate single scenes with different configs
and inspect the coarse-to-fine glimpsing behavior interactively. It runs on a few pre-uploaded samples in
[experiments/demo_samples/](experiments/demo_samples) and does **not** require installing any of
the downstream architectures.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kolner2026cfgap,
  title     = {Task-driven Processing with Coarse-to-Fine Glimpse-based Active Perception},
  author    = {Kolner, Oleh and Ortner, Thomas and Wo{\'z}niak, Stanis{\l}aw and Pantazi, Angeliki},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```

## License

Released under the OpenMDW License, version 1.0 (OpenMDW-1.0). See [LICENSE](LICENSE).
