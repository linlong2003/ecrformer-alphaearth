"""Paper-aligned SAR + cloudy-S2 ECRformer baseline on AllClear."""

from typing import Any

from .ecrformer_aef_config import ECRformerAEFConfig


class ECRformerSARAllClearConfig(ECRformerAEFConfig):
    """SAR+S2 baseline on the same original val-manifest ROI assignment.

    Assign ROIs before filtering records without Sentinel-1. This retains
    3,394 training and 763 validation samples within the optical partitions.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.dataset.kwargs.update({
            'load_aef': False,
            'load_sar': True,
            'sar_vv_min': -25.0,
            'sar_vh_min': -35.0,
            'sar_max': 0.0,
            'auto_download_missing': False,
        })

        # Paper architecture: independent 7x7 shallow stems for S1 and cloudy
        # S2, followed by channel concatenation and the ECRformer U-backbone.
        self.net.input = ['SAR', 'cloudy']
        self.net.cfg['in_chans'] = [2, self.NUM_CHANS['cloudy']]
        self.net.cfg['aef_conditioning'] = False
        self.net.cfg['decoupled_input'] = True
        self.net.cfg.pop('aef_project_channels', None)

