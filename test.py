import csv
import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import find_config_using_name
from data import find_dataset_using_name
from train import CloudRemovalModel
from util.checkpoint import (
    extract_state_dict,
    find_latest_checkpoint,
    load_checkpoint_file,
)
from util.data_split import build_eval_dataset, resolve_eval_split_name
from util.util import compute_metric


DEFAULT_DISPLAY_MIN = 0.0
DEFAULT_DISPLAY_MAX = 0.9
DEFAULT_RGB_BANDS = (3, 2, 1)  # Sentinel-2 B4/B3/B2 in the 13-band tensor.

CLOUD_CLASS_DIRS = {
    '[0,20%)': 'cloud_00_20pct',
    '[20,40%)': 'cloud_20_40pct',
    '[40,60%)': 'cloud_40_60pct',
    '[60,80%)': 'cloud_60_80pct',
    '[80,100%]': 'cloud_80_100pct',
}


def sanitize_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in name)


def resolve_log_name(config_name, run_name=None):
    return f"{config_name}_{run_name}" if run_name else config_name


def resolve_sample_name(dataset, idx: int) -> str:
    if isinstance(dataset, Subset):
        return resolve_sample_name(dataset.dataset, dataset.indices[idx])

    source = None

    if hasattr(dataset, 'paths_list'):
        source = dataset.paths_list[idx]
    elif hasattr(dataset, 'samples'):
        source = dataset.samples[idx]
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'paths'):
        source = dataset.dataset.paths[idx]

    if isinstance(source, np.ndarray):
        source = source.item() if source.ndim == 0 else source.tolist()

    if isinstance(source, dict):
        source = (source.get('sample_id') or source.get('target') or
                  source.get('cloudy') or source.get('AEF') or
                  source.get('S2') or source.get('S2_cloudy') or source.get('S1'))

    if isinstance(source, Path):
        source = str(source)
    elif isinstance(source, (list, tuple)):
        source = next((item for item in source if isinstance(item, str) and item), None)

    if isinstance(source, str) and source:
        return sanitize_name(Path(source).stem)

    return f"sample_{idx:05d}"


def resolve_sample_dates(dataset, idx: int) -> tuple[str | None, str | None]:
    """Return the actual cloudy-input and target dates for a dataset sample."""
    if isinstance(dataset, Subset):
        return resolve_sample_dates(dataset.dataset, dataset.indices[idx])

    samples = getattr(dataset, 'samples', None)
    if samples is None or idx >= len(samples) or not isinstance(samples[idx], dict):
        return None, None

    sample = samples[idx]

    def date_only(value):
        return str(value)[:10] if value else None

    return date_only(sample.get('cloudy_time')), date_only(sample.get('target_time'))


def resolve_sample_roi(dataset, idx: int) -> str | None:
    """Return the ROI identifier while preserving Subset index mapping."""
    if isinstance(dataset, Subset):
        return resolve_sample_roi(dataset.dataset, dataset.indices[idx])

    samples = getattr(dataset, 'samples', None)
    if samples is None or idx >= len(samples) or not isinstance(samples[idx], dict):
        return None
    roi = samples[idx].get('roi')
    return str(roi) if roi else None


def cloud_class_from_fraction(cloud_fraction: float) -> str:
    """Assign one image to a cloud bin from its binary cloud-mask fraction."""
    if not 0.0 <= cloud_fraction <= 1.0:
        raise ValueError(f'Cloud fraction must be in [0, 1], got {cloud_fraction}.')
    if cloud_fraction < 0.2:
        return '[0,20%)'
    if cloud_fraction < 0.4:
        return '[20,40%)'
    if cloud_fraction < 0.6:
        return '[40,60%)'
    if cloud_fraction < 0.8:
        return '[60,80%)'
    return '[80,100%]'


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['index'] = idx
        sample['sample_id'] = resolve_sample_name(self.dataset, idx)
        sample['roi'] = resolve_sample_roi(self.dataset, idx) or ''
        cloudy_date, target_date = resolve_sample_dates(self.dataset, idx)
        sample['cloudy_date'] = cloudy_date or ''
        sample['target_date'] = target_date or ''
        return sample


def get_device(gpu: int):
    if torch.cuda.is_available() and gpu >= 0:
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_model(config, ckpt_path: str, device: torch.device):
    checkpoint = load_checkpoint_file(ckpt_path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)
    model = CloudRemovalModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode normalized linear RGB values with the standard sRGB curve."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def to_rgb_image(image: np.ndarray, bands, display_min=DEFAULT_DISPLAY_MIN,
                 display_max=DEFAULT_DISPLAY_MAX):
    """Render CHW reflectance with one fixed, reproducible sRGB transform."""
    if image.ndim != 3:
        raise ValueError(f'Expected CHW image, got shape {image.shape}')
    if max(bands) >= image.shape[0]:
        raise ValueError(f'RGB bands {bands} exceed channel count {image.shape[0]}')
    if not display_max > display_min:
        raise ValueError(
            f'display_max must exceed display_min, got {display_min}, {display_max}')

    rgb = np.transpose(image[list(bands)], (1, 2, 0))
    rgb = np.nan_to_num(rgb, nan=display_min, posinf=display_max, neginf=display_min)
    linear = (rgb - display_min) / (display_max - display_min)
    srgb = linear_to_srgb(linear)
    return (srgb * 255.0).round().astype(np.uint8)


def save_prediction_npz(file_path: Path, pred: np.ndarray, index: int, sample_id: str):
    np.savez_compressed(
        file_path,
        pred=pred.astype(np.float32),
        index=np.int32(index),
        sample_id=np.array(sample_id),
    )


def save_prediction_png(file_path: Path, pred: np.ndarray, rgb_bands,
                        display_min=DEFAULT_DISPLAY_MIN,
                        display_max=DEFAULT_DISPLAY_MAX):
    Image.fromarray(to_rgb_image(
        pred, rgb_bands, display_min=display_min,
        display_max=display_max)).save(file_path)


def to_comparison_rgb_images(images, bands, display_min=DEFAULT_DISPLAY_MIN,
                             display_max=DEFAULT_DISPLAY_MAX):
    """Render every panel with identical fixed reflectance and sRGB settings."""
    return [
        to_rgb_image(
            image, bands, display_min=display_min, display_max=display_max)
        for image in images
    ]


def save_comparison_png(file_path: Path, cloudy: np.ndarray, pred: np.ndarray,
                        cloud_mask: np.ndarray, target: np.ndarray,
                        rgb_bands, display_min=DEFAULT_DISPLAY_MIN,
                        display_max=DEFAULT_DISPLAY_MAX):
    """Save a 2x2 cloudy/prediction/cloud-mask/ground-truth comparison."""
    cloudy_rgb, pred_rgb, target_rgb = to_comparison_rgb_images(
        [cloudy, pred, target], rgb_bands,
        display_min=display_min, display_max=display_max)
    if not (cloudy_rgb.shape == pred_rgb.shape == target_rgb.shape):
        raise ValueError(
            f'Comparison shape mismatch: cloudy={cloudy_rgb.shape}, '
            f'pred={pred_rgb.shape}, target={target_rgb.shape}')

    height, width = cloudy_rgb.shape[:2]
    if cloud_mask.ndim == 3 and cloud_mask.shape[0] == 1:
        cloud_mask = cloud_mask[0]
    if cloud_mask.shape != (height, width):
        raise ValueError(
            f'Cloud-mask shape mismatch: mask={cloud_mask.shape}, image={(height, width)}')
    mask_rgb = np.repeat(
        ((cloud_mask > 0.5).astype(np.uint8) * 255)[..., None], 3, axis=2)

    header_height = 28
    row_height = height + header_height
    canvas = Image.new('RGB', (width * 2, row_height * 2), 'white')
    canvas.paste(Image.fromarray(cloudy_rgb), (0, header_height))
    canvas.paste(Image.fromarray(pred_rgb), (width, header_height))
    canvas.paste(Image.fromarray(mask_rgb), (0, row_height + header_height))
    canvas.paste(Image.fromarray(target_rgb), (width, row_height + header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), 'Cloudy input', fill='black')
    draw.text((width + 8, 7), 'Prediction', fill='black')
    draw.text((8, row_height + 7), 'Cloud mask (white = cloud)', fill='black')
    draw.text((width + 8, row_height + 7), 'Ground truth', fill='black')
    draw.line((width, 0, width, row_height * 2), fill='white', width=2)
    draw.line((0, row_height, width * 2, row_height), fill='white', width=2)
    canvas.save(file_path)


def write_metrics_csv(file_path: Path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with file_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(file_path: Path, summary):
    with file_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def build_dataloader(config, split_override=None, batch_size=None, num_workers=None, max_samples=None):
    dataset_class = find_dataset_using_name(config.dataset.name)
    split_name, dataset = build_eval_dataset(
        config,
        dataset_class,
        split_override=split_override,
    )

    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    dataset = IndexedDataset(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size or config.train.valid_bs,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers if num_workers is not None else config.train.num_workers) > 0,
    )
    return split_name, dataset, loader


def run_evaluation(args):
    if (args.group_comparisons_by_cloud and
            args.export_format not in {'png', 'both'}):
        raise ValueError(
            '--group-comparisons-by-cloud requires --export-format png or both.')

    config_class = find_config_using_name(args.config)
    config = config_class()
    if args.legacy_val_holdout:
        if args.split is not None:
            raise ValueError('--legacy-val-holdout cannot be combined with --split.')
        # Reproduce checkpoints trained with the historical ROI-grouped 80/20
        # split of the AllClear val branch. Full-val evaluation would include
        # their training ROIs and cause geographic leakage.
        config.dataset.split = ['val']
    config.name = args.name
    config.train.gpu = [args.gpu]
    if (args.export_format in {'png', 'both'} and
            config.dataset.name == 'allclear_aef'):
        config.dataset.kwargs = dict(getattr(config.dataset, 'kwargs', {}))
        config.dataset.kwargs['include_cloud_mask'] = True

    split = resolve_eval_split_name(config, args.split)
    log_name = resolve_log_name(args.config, args.name)
    output_dir = Path(args.output_dir or Path('results') / log_name / split)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation split: {split}")
    print(f"Output directory: {output_dir}")

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path, _ = find_latest_checkpoint(config.train.save_dir, log_name)
    if ckpt_path is None:
        raise FileNotFoundError('Checkpoint not found. Please specify one with --ckpt-path.')
    print(f"Using checkpoint: {ckpt_path}")

    device = get_device(args.gpu)
    _, _, dataloader = build_dataloader(
        config,
        split_override=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataloader.dataset)} samples for evaluation.")
    model = load_model(config, ckpt_path, device)

    npz_dir = output_dir / 'pred_npz'
    png_dir = output_dir / 'pred_rgb'
    comparison_dir = output_dir / 'compre'
    cloud_group_counts = Counter()
    if args.group_comparisons_by_cloud:
        print(
            'Grouping each comparison image by its own binary cloud-mask fraction.')
    if args.export_format in {'npz', 'both'}:
        npz_dir.mkdir(parents=True, exist_ok=True)
    if args.export_format in {'png', 'both'}:
        png_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)
        if args.group_comparisons_by_cloud:
            for directory in CLOUD_CLASS_DIRS.values():
                (comparison_dir / directory).mkdir(parents=True, exist_ok=True)

    metric_rows = []
    metric_names = ['RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM', 'LPIPS']
    summary_accumulator = {name: [] for name in metric_names}
    cloud_metric_accumulator = {
        label: {name: [] for name in metric_names}
        for label in CLOUD_CLASS_DIRS
    } if args.group_comparisons_by_cloud else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = move_batch_to_device(batch, device)
            _, model_input, target = model.fuse_input(batch)
            pred_result = model(model_input)
            pred = pred_result[0] if isinstance(pred_result, tuple) else pred_result

            metrics = compute_metric(pred, target, size_average=False)
            pred_np = tensor_to_numpy(pred)
            cloudy_np = tensor_to_numpy(batch['cloudy'])
            target_np = tensor_to_numpy(target)
            cloud_mask_np = (
                tensor_to_numpy(batch['cloud_mask']) if 'cloud_mask' in batch else None)
            indices = batch['index'].detach().cpu().tolist()
            sample_ids = batch['sample_id']
            sample_rois = batch['roi']
            cloudy_dates = batch['cloudy_date']
            target_dates = batch['target_date']
            batch_cloud_fractions = None
            batch_cloud_classes = None
            if args.group_comparisons_by_cloud:
                if cloud_mask_np is None:
                    raise KeyError(
                        'Cloud grouping requires batch["cloud_mask"].')
                batch_cloud_fractions = [
                    float(np.mean(mask > 0.5)) for mask in cloud_mask_np
                ]
                batch_cloud_classes = [
                    cloud_class_from_fraction(value)
                    for value in batch_cloud_fractions
                ]

            for item_idx, (sample_idx, sample_id, sample_roi, cloudy_date, target_date,
                           sample_cloudy, sample_pred, sample_target) in enumerate(zip(
                    indices, sample_ids, sample_rois, cloudy_dates, target_dates,
                    cloudy_np, pred_np, target_np)):
                if cloudy_date and target_date:
                    roi = sample_id.split('_', 1)[0]
                    file_stem = (
                        f"{sample_idx:05d}_{sanitize_name(roi)}_"
                        f"cloudy_{sanitize_name(cloudy_date)}_"
                        f"target_{sanitize_name(target_date)}")
                else:
                    file_stem = f"{sample_idx:05d}_{sanitize_name(sample_id)}"
                if args.export_format in {'npz', 'both'}:
                    save_prediction_npz(npz_dir / f'{file_stem}.npz', sample_pred, sample_idx, sample_id)
                if args.export_format in {'png', 'both'}:
                    if cloud_mask_np is None:
                        raise KeyError(
                            'PNG comparison export requires batch["cloud_mask"].')
                    save_prediction_png(
                        png_dir / f'{file_stem}.png',
                        sample_pred,
                        args.rgb_bands,
                        display_min=args.display_min,
                        display_max=args.display_max,
                    )
                    sample_comparison_dir = comparison_dir
                    if args.group_comparisons_by_cloud:
                        cloud_class = batch_cloud_classes[item_idx]
                        sample_comparison_dir = (
                            comparison_dir / CLOUD_CLASS_DIRS[cloud_class])
                        cloud_group_counts[cloud_class] += 1
                    save_comparison_png(
                        sample_comparison_dir / f'{file_stem}.png',
                        sample_cloudy,
                        sample_pred,
                        cloud_mask_np[item_idx],
                        sample_target,
                        args.rgb_bands,
                        display_min=args.display_min,
                        display_max=args.display_max,
                    )

            batch_size = pred.shape[0]
            for item_idx in range(batch_size):
                row = {
                    'index': int(indices[item_idx]),
                    'sample_id': sample_ids[item_idx],
                    'roi': sample_rois[item_idx],
                }
                cloud_class = None
                if args.group_comparisons_by_cloud:
                    cloud_fraction = batch_cloud_fractions[item_idx]
                    cloud_class = batch_cloud_classes[item_idx]
                    row['cloud_fraction'] = cloud_fraction
                    row['cloud_percent'] = 100.0 * cloud_fraction
                    row['cloud_class'] = cloud_class
                for key in metric_names:
                    row[key] = float(metrics[key][item_idx].item())
                metric_rows.append(row)
                for key in summary_accumulator:
                    summary_accumulator[key].append(row[key])
                    if cloud_class is not None:
                        cloud_metric_accumulator[cloud_class][key].append(row[key])

    summary = {
        'config': args.config,
        'run_name': args.name,
        'split': split,
        'legacy_val_holdout': bool(args.legacy_val_holdout),
        'evaluation_protocol': (
            'ROI-grouped 80/20 holdout of the AllClear val branch (seed 42)'
            if args.legacy_val_holdout else
            'configured dataset evaluation branch'
        ),
        'checkpoint': str(ckpt_path),
        'num_samples': len(metric_rows),
        'metrics': {
            key: float(np.mean(values)) if values else None
            for key, values in summary_accumulator.items()
        },
        'export_format': args.export_format,
        'rgb_bands': list(args.rgb_bands),
        'display_transform': 'fixed linear reflectance to standard sRGB',
        'display_min': float(args.display_min),
        'display_max': float(args.display_max),
        'comparison_dir': str(comparison_dir) if args.export_format in {'png', 'both'} else None,
        'cloud_grouping_method': (
            'per-image fraction of cloud_mask pixels > 0.5'
            if args.group_comparisons_by_cloud else None),
        'comparison_cloud_group_counts': {
            label: int(cloud_group_counts[label]) for label in CLOUD_CLASS_DIRS
        } if args.group_comparisons_by_cloud else None,
        'metrics_by_cloud_class': {
            label: {
                'num_samples': len(cloud_metric_accumulator[label][metric_names[0]]),
                'metrics': {
                    key: (float(np.mean(values)) if values else None)
                    for key, values in cloud_metric_accumulator[label].items()
                },
            }
            for label in CLOUD_CLASS_DIRS
        } if cloud_metric_accumulator is not None else None,
    }

    write_metrics_csv(output_dir / 'metrics.csv', metric_rows)
    write_summary_json(output_dir / 'summary.json', summary)

    print(f"Saved metrics to: {output_dir / 'metrics.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='ecrformer_aef', help='Config name')
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--split', type=str, default=None, help='Dataset split to evaluate')
    parser.add_argument(
        '--legacy-val-holdout', action='store_true',
        help=(
            'Reproduce the historical ROI-grouped 80/20 holdout of the '
            'AllClear val branch; do not combine with --split.'))
    parser.add_argument('--ckpt-path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for evaluation outputs')
    parser.add_argument('--export-format', choices=['none', 'npz', 'png', 'both'], default='both',
                        help='Prediction export format')
    parser.add_argument('--rgb-bands', nargs=3, type=int, default=DEFAULT_RGB_BANDS,
                        help='RGB band indices for PNG export')
    parser.add_argument('--display-min', type=float, default=DEFAULT_DISPLAY_MIN,
                        help='Fixed reflectance mapped to sRGB black')
    parser.add_argument('--display-max', type=float, default=DEFAULT_DISPLAY_MAX,
                        help='Fixed reflectance mapped to sRGB white')
    parser.add_argument('--batch-size', type=int, default=None, help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of dataloader workers')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit the number of evaluated samples')
    parser.add_argument(
        '--group-comparisons-by-cloud', action='store_true',
        help=('Save comparison PNGs into five per-image cloud-amount '
              'subdirectories, using each sample cloud mask'))
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index, use a negative value for CPU')
    return parser.parse_args()


if __name__ == '__main__':
    run_evaluation(parse_args())
