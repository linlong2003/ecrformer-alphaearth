"""SAR baseline with the historical, unfiltered AllClear-val ROI assignment."""

from .ecrformer_sar_allclear_config import ECRformerSARAllClearConfig


class ECRformerSARLegacyConfig(ECRformerSARAllClearConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset.split = ['val']
        self.dataset.train_ratio = 0.8
        self.seed = 42
        self.dataset.kwargs['preserve_manifest_roi_split'] = True
