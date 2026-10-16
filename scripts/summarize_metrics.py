"""Summarize an evaluation CSV on fixed paper validation IDs; no inference."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM', 'LPIPS')


def metric_id(sample_id):
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_'
                   for ch in Path(sample_id).stem)


def summarize(metrics_path, partition='validation', snapshot_path=None):
    snapshot_path = snapshot_path or ROOT / 'splits' / 'allclear_val_roi80_seed42.json'
    snapshot = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
    selected = snapshot['partitions'][partition]
    ids = [metric_id(sid) for sid in selected['sample_ids']]
    if len(set(ids)) != len(ids):
        raise ValueError('Snapshot sample IDs are not unique after evaluator normalization')
    with Path(metrics_path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row['sample_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate sample IDs in metrics CSV')
    if set(by_id) - {metric_id(sid) for sid in snapshot['partitions']['validation']['sample_ids']}:
        raise ValueError('CSV contains samples outside the paper validation holdout')
    if set(ids) - by_id.keys():
        raise ValueError(f'CSV is missing {len(set(ids) - by_id.keys())} required validation samples')
    selected_rows = [by_id[sid] for sid in ids]
    if any(row['roi'] not in selected['roi_ids'] for row in selected_rows):
        raise ValueError('CSV ROI is outside the selected validation partition')
    result = {'num_samples': len(ids), 'partition': partition, 'metrics': {}}
    for metric in METRICS:
        values = [float(row[metric]) for row in selected_rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'Nonfinite {metric}')
        result['metrics'][metric] = mean(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True, type=Path)
    parser.add_argument('--partition', choices=('validation', 'sar_validation'), default='validation')
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.metrics, args.partition, args.snapshot), indent=2))


if __name__ == '__main__':
    main()
