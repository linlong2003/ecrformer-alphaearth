"""Zero-information AEF control, preserving the complete Proposed network."""
from .ecrformer_aef_config import ECRformerAEFConfig


class ECRformerAEFZeroLegacyConfig(ECRformerAEFConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = 42
        self.dataset.split = ['val']
        self.dataset.train_ratio = 0.8
        self.dataset.kwargs.update(zero_aef=True, load_aef=True, load_sar=False,
                                   preserve_manifest_roi_split=True,
                                   auto_download_missing=False)
        self.train.ckpt_path = None
