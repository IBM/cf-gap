import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO
DATA_PATH_HR_INSDET = ''
DATA_PATH_ROBOTOOLS = ''
CHECKPOINT_PATH_FINE_SEARCH_MAP = os.path.join(ROOT_DIR, 'fine_search_map_checkpoint.pt')
CHECKPOINT_PATH_STT = ''

CHECKPOINT_PATH_GROUNDING_DINO = ''
CONFIG_PATH_GROUNDING_DINO = ''

CHECKPOINT_PATH_MOBILESAM = ''

# NIDS-Net downstream architecture
CHECKPOINT_PATH_SAM = ''            # SAM ViT-T checkpoint (loaded via mobile_sam.sam_model_registry)
CHECKPOINT_PATH_NIDSNET_ADAPTER = ''  # WeightAdapter weights for NIDS-Net feature matching
