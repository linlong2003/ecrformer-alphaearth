# Third-party notices and modification record

## ECRformer

This project adapts ECRformer rather than claiming the reconstruction backbone,
attention blocks, or semantic-decoupled feature supervision as new work.

- Source: https://github.com/zzaiyan/ECRformer
- Authors: Zaiyan Zhang, Jie Li, Yuanqi Liang, Jining Yan, Yi Xiao, Xin Su,
  and Qiangqiang Yuan.
- Paper: *ECRformer: An efficient cloud removal transformer with
  semantic-decoupled learning for multimodal satellite imagery*, ISPRS Journal
  of Photogrammetry and Remote Sensing, vol. 237, pp. 323–338, 2026.
- DOI: https://doi.org/10.1016/j.isprsjprs.2026.04.009

The retained implementation includes the model and attention modules,
configuration and dataset utilities, training/evaluation infrastructure,
checkpoint handling, and SSIM implementation. Original notices within copied
source files are retained. Local modifications add AEF adaptation and gated
conditioning, the AllClear adapter, paper-protocol configurations, FP32
checks, and reproducibility utilities.

On 2026-10-16, the upstream LICENSE was verified as the GNU General Public
License, version 3. Its Git blob SHA was
`19ef76772b59ad4665c3fadf9197cb685e227959`.
The full upstream license text is retained in [LICENSE](LICENSE).
This modified work follows GPL version 3; no MIT/Apache relicensing of the
ECRformer-derived work is claimed. Upstream and third-party rights remain
with their respective holders.

Modification record (2026-10-16 preparation snapshot):

- `models/`: AEF adaptation and multiscale gated conditioning added to the
  ECRformer backbone; upstream attention and backbone components retained.
- `data/` and `config/`: AllClear/AEF loading, paper split settings, and
  environment-configurable paths; automatic missing-data downloads disabled.
- `train.py`, `test.py`, and `util/`: training/evaluation integration,
  checkpoint handling, and numerical safeguards.
- `scripts/`, `splits/`, `tests/`, and documentation: release preparation,
  protocol identifiers, and verification utilities. Tests have not been
  completed as part of this preparation stage.

## Restormer

Layer-normalization helpers in `models/module_util.py` reference Restormer.

- Source: https://github.com/swz30/Restormer
- License: MIT.
- Copyright (c) 2022 Syed Waqas Zamir and contributors.
- Full notice: [licenses/Restormer-MIT.txt](licenses/Restormer-MIT.txt).

The MIT notice applies to the corresponding third-party portions and is
preserved alongside the GPL-3.0 license for the combined derivative work.

## DRSformer: remaining provenance review

`TopkAttention` in `models/module.py` explicitly references
https://github.com/cschenxiang/DRSformer. The upstream `setup.py` declares
Apache-2.0 using BasicSR project metadata, but the scope of that declaration
for DRSformer-specific additions has not been independently confirmed.
The proposed TSA configuration does not select TopkAttention, but its source
is still present. Do not treat ECRformer's new license as independent proof
of rights in all embedded third-party code. Resolve this scope before a
public release; no separate authorization from DRSformer is claimed here.

## pytorch-ssim: remaining notice review

`util/pytorch_ssim/__init__.py` closely corresponds to the implementation at
https://github.com/Po-Hsun-Su/pytorch-ssim, with local numerical changes.
That upstream repository's `LICENSE.txt` contains the single word `MIT`.
Confirm the appropriate copyright attribution and complete notice before
public release; no missing copyright statement has been invented here.

## Data and dependencies

AllClear data, AlphaEarth Foundations embeddings, pretrained weights, and
third-party baseline source trees are not bundled. Obtain them from their
respective providers and follow their terms. Installed Python dependencies
retain their respective licenses. Do not upload credentials, private data,
or restricted third-party paper PDFs with this repository.
