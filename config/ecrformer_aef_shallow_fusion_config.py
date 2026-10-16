"""Controlled unit-normalized AEF shallow-fusion ablation."""

from typing import Any

from .ecrformer_aef_config import ECRformerAEFConfig


class ECRformerAEFShallowFusionConfig(ECRformerAEFConfig):
    """Use the original ECRformer dual-stem fusion without AEF conditioning."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Preserve separate shallow embeddings for AEF and cloudy S2, then
        # concatenate their feature maps as in the original multimodal stem.
        self.net.cfg['aef_conditioning'] = False
        self.net.cfg['decoupled_input'] = True
        self.net.cfg.pop('aef_project_channels', None)
