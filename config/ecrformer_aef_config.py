import os
from pathlib import Path
from typing import Any

from .ecrformer_config import ECRformerConfig


class ECRformerAEFConfig(ECRformerConfig):
    """Paper ROI-disjoint AllClear holdout with 64-channel AEF input."""

    NUM_CHANS = {
        # Kept for parent-config initialization; not used by this config's net input.
        'SAR': 2,
        'AEF': 64,
        'cloudy': 13,
        'target': 13,
    }

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        datasets_root = Path(os.environ.get(
            'ECRFORMER_DATASETS_ROOT',
            str(Path(__file__).resolve().parents[1] / 'datasets'),
        )).expanduser()
        allclear_root = Path(os.environ.get(
            'ECRFORMER_ALLCLEAR_ROOT',
            str(datasets_root / 'Allclear'),
        )).expanduser()
        shared_aef_root = os.environ.get('ECRFORMER_AEF_ROOT')
        test_aef_root = Path(os.environ.get(
            'ECRFORMER_TEST_AEF_ROOT',
            shared_aef_root or str(datasets_root / 'test_aef_downloads_cropped'),
        )).expanduser()
        val_aef_root = Path(os.environ.get(
            'ECRFORMER_VAL_AEF_ROOT',
            shared_aef_root or str(datasets_root / 'val_aef_downloads_cropped'),
        )).expanduser()

        self.dataset.name = 'allclear_aef'
        self.dataset.root = str(allclear_root)
        # Paper protocol: split the original val manifest by ROI, not by sample.
        # This is an internal holdout, not the official AllClear benchmark split.
        self.dataset.split = ['val']
        self.dataset.train_ratio = 0.8
        self.seed = 42
        # The paper samples a 256x256 source window, then applies a random
        # 128x128 crop during training.
        self.dataset.crop_size = 128
        self.dataset.kwargs = {
            'manifest_names': {
                'test': 'test_tx3_s2-s1_100pct.json',
                'val': 'val_tx12_s2-s1_100pct.json',
            },
            'data_subdirs': {'test': 'test', 'val': 'val'},
            'aef_roots': {
                'test': str(test_aef_root),
                'val': str(val_aef_root),
            },
            'aef_year': 2023,
            'output_size': 256,
            's2_max': 10000.0,
            'preserve_manifest_roi_split': True,
            'auto_download_missing': False,
            'downloader_path': str(allclear_root / 'download.py'),
            'download_timeout': 3600,
        }

        self.net.input = ['AEF', 'cloudy']
        self.net.output = ['target']
        self.net.cfg['in_chans'] = [self.NUM_CHANS[key] for key in self.net.input]
        self.net.cfg['out_chans'] = sum(self.NUM_CHANS[key] for key in self.net.output)
        self.net.cfg['aef_conditioning'] = True
        self.net.cfg['aef_project_channels'] = 16

        # Paper training hyperparameters. Model architecture and checkpoint keys
        # are unchanged from the development implementation.
        self.train.train_bs = 4
        self.train.valid_bs = 4
        self.train.num_workers = 2
        self.train.compute_lpips = False
        self.optim.accumulate_grad_batches = 4
        self.optim.precision = '32-true'
        self.train.lr = 4e-4
        # Original SDFL coefficients: L = L_output + 0.05 R_enc + 0.05 R_dec.
        self.train.proj_weight = [0.05, 0.05]
