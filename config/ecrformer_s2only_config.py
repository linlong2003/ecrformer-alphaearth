"""Controlled Sentinel-2-only ablation for the AllClear AEF experiment."""

from typing import Any

from .ecrformer_aef_config import ECRformerAEFConfig


class ECRformerS2OnlyConfig(ECRformerAEFConfig):
    """Match the AEF run while removing every AEF input and model branch."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Keep the identical manifest, ROI split and S2 preprocessing, but do
        # not open or return an AEF raster from the dataset.
        self.dataset.kwargs['load_aef'] = False

        # Use exactly the same 13 -> 48 visual stem as the conditioned model's
        # primary path. No AEF adapter, projection, gate or decoupled encoder is
        # instantiated in this controlled baseline.
        self.net.input = ['cloudy']
        self.net.cfg['in_chans'] = [self.NUM_CHANS['cloudy']]
        self.net.cfg['aef_conditioning'] = False
        self.net.cfg['decoupled_input'] = False
        self.net.cfg.pop('aef_project_channels', None)
