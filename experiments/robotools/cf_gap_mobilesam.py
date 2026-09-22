import os.path
import torchvision
import torch
from torch import nn

from coarse_to_fine_gap import CoarseToFineGAP
from fine_glimpsing.fine_gap import FineGAP
from fine_glimpsing import fine_search_map
from fine_glimpsing import basic_modules
from fine_glimpsing.logpolar_sensor import LogPolarSensor
from coarse_map import CoarseSearchMapGeneration, IoRMasker
from fine_glimpsing.target_glimpse_generation import TargetGlimpseExtractor
from eval_datasets.dataloaders import Robotools
import project_utils
import project_definitions
from experiments.evaluation import evaluate

torch.autograd.set_grad_enabled(False)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 123
project_utils.set_global_seed(seed)

SCENE_SIZE = [1080, 1920]
SEARCH_TARGET_SIZE = 512       # size of search target examples (squared images)


def get_dataset():
    dataset = Robotools(
        root_dir_scenes=os.path.join(project_definitions.DATA_PATH_ROBOTOOLS, 'test'),
        root_dir_objects=os.path.join(project_definitions.DATA_PATH_ROBOTOOLS, 'test_video'),
        resize_targets=SEARCH_TARGET_SIZE,
        every_n_example=4,  # for faster experimentation, setting to None would give better result
    )
    return dataset


def get_model(with_downstream_arch=True):
    # --- Initialize fine search map generator
    glimpse_size = (245, 245)       # size of log-polar image
    feat_dim = 96
    n_feats = 128   # number of vectors in external embeddings
    h_rho, w_phi = glimpse_size
    patch_size = 5
    n_heads = 4

    dummy_input = torch.randn(5, 3, h_rho, w_phi)
    patch_emb = nn.Conv2d(3, feat_dim, kernel_size=patch_size, stride=patch_size)
    feature_grid_size = tuple(patch_emb(dummy_input).shape[-2:])

    scene_encoder = basic_modules.BottomUpTopDownAttention(
        n_feats, feat_dim,
        bu_attn=basic_modules.CrossAttentionBlock(
            feat_dim, n_heads, share_norm='qkv'
        ),
        td_attn=basic_modules.CrossAttentionBlock(
            feat_dim, n_heads, share_norm='qkv'
        ),
        patch_encoder=patch_emb,
        feat_self_attention=basic_modules.CrossAttentionBlock(
            feat_dim, n_heads, share_norm='qkv'
        ),
        use_external_features_td_values=True, grid_size=feature_grid_size,
        max_feat_norm=1.,
        use_lnorm_external_features=True,
        use_instance_norm=True
    )
    search_target_encoder = scene_encoder

    feature_comparison = fine_search_map.FeatureCorrelator()
    fine_search_map_generator = fine_search_map.FineSearchMapGeneration(
        scene_encoder, search_target_encoder, feature_comparison)

    fine_search_map_generator.load_state_dict(torch.load(project_definitions.CHECKPOINT_PATH_FINE_SEARCH_MAP))
    fine_search_map_generator = fine_search_map.FineSearchMapGenerationWrapper(fine_search_map_generator, every_n_example=4)

    sensor = LogPolarSensor(glimpse_size, radius=512, skew=1.)

    # radius is smaller since search target images are much smaller (for faster experimentation)
    search_target_sensor = LogPolarSensor(glimpse_size, radius=512, skew=1.)

    # --- Initialize coarse search map generator
    pretrained = torchvision.models.mobilenet_v3_large(pretrained=True)
    pretrained.eval()
    normalizer = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    coarse_search_map_extractor = CoarseSearchMapGeneration(
        nn.Sequential(normalizer, nn.Sequential(*list(pretrained.children())[0][:-5]), nn.InstanceNorm2d(112)),
        norm_type_target='l2',
    )
    dummy_map = coarse_search_map_extractor(
        torch.zeros(1, 3, *SCENE_SIZE), torch.zeros(1, 1, 3, SEARCH_TARGET_SIZE, SEARCH_TARGET_SIZE))
    coarse_map_size = list(dummy_map.shape[-2:])

    # --- Initialize downstream architecture
    if with_downstream_arch:
        import downstream_architectures
        embedder_core = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        embedder_core.eval()

        # used to extract features of candidate region and match them to examples of the search targets
        visual_embedder = nn.Sequential(
            project_utils.Lambda(lambda x: project_utils.auto_adjust_contrast(x)),
            normalizer,
            embedder_core,
        )
        visual_embedder.eval()
        downstream_architecture = downstream_architectures.SAMDownstreamArch(
            n_prompt_locs=3,  # same as number of fine glimpses
            early_stopping_threshold=1.2,  # ignored
            crop_to_scene_size_ratio=0.35,
            visual_embedder=visual_embedder,
            use_big_sam=False,
            checkpoint_path=project_definitions.CHECKPOINT_PATH_MOBILESAM
        )
    else:
        downstream_architecture = None

    # --- Initialize fine glimpsing
    fine_gap = FineGAP(
        sensor, fine_search_map_generator,
        n_glimpses=1,
        temperature=10,
        downstream_architecture=downstream_architecture,
    )

    # --- Initialize coarse-to-fine GAP
    model = CoarseToFineGAP(
        coarse_map_extractor=coarse_search_map_extractor,
        n_coarse_glimpses=30,
        fine_glimpsing=fine_gap,
        ior_masker=IoRMasker(coarse_map_size, eps=2.),
        ior_coarse_glimpses_only=True,
        search_target_glimpses_extraction=TargetGlimpseExtractor(search_target_sensor, end_with_central_glimpse=True)
    )
    model.eval()
    model.to(DEVICE)
    return model


if __name__ == '__main__':
    _dataset = get_dataset()
    _model = get_model()
    evaluate(_model, _dataset, DEVICE)
