"""Verify the paper's fixed ROI partitions using manifest metadata, not rasters."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.data_split import build_group_split_indices

DEFAULT_SNAPSHOT = ROOT / 'splits' / 'allclear_val_roi80_seed42.json'
EXPECTED_COUNTS = {
    'train': (4510, 787), 'validation': (1123, 197),
    'sar_train': (3394, 626), 'sar_validation': (763, 145),
}


def build_snapshot(manifest_path):
    manifest_path = Path(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    records = list(manifest.items())
    groups = [record['roi'][0] for _, record in records]
    train, valid = build_group_split_indices(groups, 0.8, 42)
    partitions = {}
    for name, indices in (('train', train), ('validation', valid)):
        for sar_only in (False, True):
            selected = [(sid, record) for i in indices for sid, record in [records[i]]
                        if not sar_only or record.get('s1')]
            key = 'sar_' + name if sar_only else name
            partitions[key] = {
                'sample_ids': [sid for sid, _ in selected],
                'roi_ids': list(dict.fromkeys(record['roi'][0] for _, record in selected)),
            }
    return {
        'protocol': 'Internal ROI-disjoint 80:20 split of the AllClear val branch',
        'manifest': manifest_path.name,
        'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'seed': 42, 'train_ratio': 0.8,
        'roi_order': 'First occurrence in the original manifest before SAR filtering',
        'partitions': partitions,
    }


def check_expected_counts(snapshot):
    for name, (samples, rois) in EXPECTED_COUNTS.items():
        partition = snapshot['partitions'][name]
        actual = len(partition['sample_ids']), len(partition['roi_ids'])
        if actual != (samples, rois):
            raise ValueError(f'{name}: expected {(samples, rois)} samples/ROIs, got {actual}')
    partitions = snapshot['partitions']
    if set(partitions['train']['roi_ids']) & set(partitions['validation']['roi_ids']):
        raise ValueError('Training and validation ROIs overlap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--write-snapshot', type=Path,
                        help='Maintainer-only export to a NEW file; never overwrites a snapshot.')
    args = parser.parse_args()
    actual = build_snapshot(args.manifest)
    check_expected_counts(actual)
    if args.write_snapshot:
        args.write_snapshot.parent.mkdir(parents=True, exist_ok=True)
        with args.write_snapshot.open('x', encoding='utf-8') as handle:
            json.dump(actual, handle, indent=2)
            handle.write('\n')
    else:
        expected = json.loads(args.snapshot.read_text(encoding='utf-8'))
        if actual != expected:
            raise ValueError('Manifest hash, ordering, or partition membership differs from the paper snapshot. '
                             'Use the original manifest; do not overwrite the snapshot to bypass this check.')
    for name, partition in actual['partitions'].items():
        print(f'{name}: {len(partition["sample_ids"])} samples / {len(partition["roi_ids"])} ROIs')
    print('PASS: fixed ROI membership, SAR filtering, and manifest checksum verified; no raster reads.')


if __name__ == '__main__':
    main()
