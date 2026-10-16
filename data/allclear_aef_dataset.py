"""Single-frame AllClear optical + AEF/SAR dataset for ECRformer.

The AllClear tx12 manifest defines a target and twelve Sentinel-2
observations.  This adapter deterministically selects the observation nearest
to the target time and replaces ECRformer's SAR input with the 64-band AEF
embedding for the sample ROI.

The same adapter can expose the nearest Sentinel-1 observation for a
paper-aligned optical-SAR ECRformer baseline. Samples without Sentinel-1 are
excluded when ``load_sar=True``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _nearest_s2(target_time: str, observations: list[list[str]]) -> list[str]:
    if not observations:
        raise ValueError('AllClear sample has no s2_toa observations.')
    target = _parse_time(target_time)
    return min(observations, key=lambda item: abs(_parse_time(item[0]) - target))


def _nearest_s1(target_time: str, observations: list[list[str]]) -> list[str]:
    if not observations:
        raise ValueError('AllClear sample has no s1 observations.')
    target = _parse_time(target_time)
    return min(observations, key=lambda item: abs(_parse_time(item[0]) - target))


class _RasterAccessError(RuntimeError):
    """Attach the failing raster path to a Rasterio read/open error."""

    def __init__(self, path: Path, cause: Exception):
        super().__init__(f'Cannot read raster {path}: {cause}')
        self.path = path
        self.cause = cause


class AllClearAEFDataset(Dataset):
    """Return aligned AllClear tensors, optionally including the AEF prior."""

    def __init__(
        self,
        root,
        split='val',
        data_range=1.0,
        crop_size=None,
        manifest_dir=None,
        manifest_name=None,
        manifest_names=None,
        data_subdirs=None,
        aef_root=None,
        aef_roots=None,
        aef_year=2023,
        output_size=256,
        s2_max=10000.0,
        auto_download_missing=False,
        downloader_path=None,
        download_timeout=3600,
        include_cloud_mask=False,
        load_aef=True,
        load_sar=False,
        sar_vv_min=-25.0,
        sar_vh_min=-35.0,
        sar_max=0.0,
        preserve_manifest_roi_split=False,
        zero_aef=False,
    ):
        self.root = Path(root)
        self.split = str(split)
        data_subdirs = dict(data_subdirs or {})
        self.data_root = self.root / data_subdirs.get(self.split, self.split)
        self.manifest_dir = (Path(manifest_dir) if manifest_dir else
                             self.root / 'metadata' / 'datasets')
        aef_roots = dict(aef_roots or {})
        selected_aef_root = aef_roots.get(self.split, aef_root)
        self.aef_root = (Path(selected_aef_root) if selected_aef_root else
                         self.root.parent / f'{self.split}_aef_downloads_cropped')
        self.aef_year = int(aef_year)
        self.data_range = float(data_range)
        self.crop_size = int(crop_size) if crop_size is not None else None
        self.output_size = int(output_size) if output_size is not None else None
        self.s2_max = float(s2_max)
        self.auto_download_missing = bool(auto_download_missing)
        self.downloader_path = (Path(downloader_path) if downloader_path else
                                self.root / 'download.py')
        self.download_timeout = int(download_timeout)
        self.include_cloud_mask = bool(include_cloud_mask)
        self.load_aef = bool(load_aef)
        self.zero_aef = bool(zero_aef)
        if self.zero_aef and not self.load_aef:
            raise ValueError('zero_aef requires load_aef=True; keep the AEF branch')
        self.load_sar = bool(load_sar)
        self.sar_db_mins = np.asarray(
            [float(sar_vv_min), float(sar_vh_min)], dtype=np.float32)[:, None, None]
        self.sar_max = float(sar_max)
        if np.any(self.sar_db_mins >= self.sar_max):
            raise ValueError(
                f'SAR minima must be below sar_max={self.sar_max}: '
                f'{self.sar_db_mins[:, 0, 0].tolist()}')
        self.download_lock_dir = self.root / '.download_locks'

        manifest_names = dict(manifest_names or {})
        selected_manifest = manifest_names.get(self.split, manifest_name)
        if selected_manifest is None:
            candidates = [
                f'{self.split}_tx12_s2-s1_100pct.json',
                f'{self.split}_tx3_s2-s1_100pct.json',
            ]
            selected_manifest = next(
                (name for name in candidates
                 if (self.manifest_dir / name).is_file()),
                candidates[0],
            )
        else:
            selected_manifest = selected_manifest.format(split=self.split)
        manifest_path = self.manifest_dir / selected_manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f'AllClear manifest not found: {manifest_path}')
        with manifest_path.open('r', encoding='utf-8') as handle:
            manifest = json.load(handle)

        if preserve_manifest_roi_split:
            # Keep the original ROI ordering before modality filtering.
            self.split_reference_groups = [record['roi'][0] for record in manifest.values()]

        self.samples = []
        for sample_id, record in manifest.items():
            roi = record['roi'][0]
            target_time, target_rel = record['target'][0]
            cloudy_time, cloudy_rel = _nearest_s2(target_time, record['s2_toa'])
            s1_observations = record.get('s1') or []
            if self.load_sar and not s1_observations:
                continue
            if self.load_sar:
                sar_time, sar_rel = _nearest_s1(target_time, s1_observations)
            cloudy_rel_path = Path(cloudy_rel)
            mask_name = cloudy_rel_path.name.replace('_s2_toa_', '_cld_shdw_')
            mask_rel = cloudy_rel_path.parent.parent / 'cld_shdw' / mask_name
            sample = {
                'sample_id': sample_id,
                'roi': roi,
                'AEF': self.aef_root / f'{roi}_{self.aef_year}_utm_cropped.tif',
                'cloudy': self.data_root / cloudy_rel,
                'cloud_mask': self.data_root / mask_rel,
                'target': self.data_root / target_rel,
                'cloudy_time': cloudy_time,
                'target_time': target_time,
            }
            if self.load_sar:
                sample['SAR'] = self.data_root / sar_rel
                sample['sar_time'] = sar_time
            self.samples.append(sample)
        self.split_groups = [sample['roi'] for sample in self.samples]
        if self.load_aef:
            required_aef = list(dict.fromkeys(
                sample['AEF'] for sample in self.samples))
            missing_aef = [path for path in required_aef if not path.is_file()]
            if missing_aef:
                examples = '\n'.join(str(path) for path in missing_aef[:5])
                raise FileNotFoundError(
                    f'AEF coverage check failed for split {self.split!r}: '
                    f'{len(missing_aef)} of {len(required_aef)} required ROI AEF '
                    f'rasters are missing under {self.aef_root}. First missing paths:\n'
                    f'{examples}\nSet the split-specific AEF root in the '
                    f'configuration after downloading/generating those embeddings.')

    def __len__(self):
        return len(self.samples)

    @contextmanager
    def _roi_download_lock(self, roi: str):
        """Cross-worker lock preventing duplicate downloads of one ROI."""
        self.download_lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.download_lock_dir / f'{roi}.lock'
        deadline = time.monotonic() + self.download_timeout
        acquired = False
        while not acquired:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    handle.write(f'pid={os.getpid()}\ntime={time.time()}\n')
                acquired = True
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                    if age > self.download_timeout:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'Timed out waiting for automatic download lock: {lock_path}')
                time.sleep(2.0)
        try:
            yield
        finally:
            if acquired:
                lock_path.unlink(missing_ok=True)

    def _run_roi_download(self, sample, affected_paths, reason, previous_signatures=None):
        roi = sample['roi']
        if not self.auto_download_missing:
            paths = ', '.join(str(path) for path in affected_paths)
            raise RuntimeError(f'AllClear raster recovery is disabled ({reason}): {paths}')
        if not self.downloader_path.is_file():
            raise FileNotFoundError(
                f'AllClear downloader not found: {self.downloader_path}')

        with self._roi_download_lock(roi):
            if previous_signatures is not None:
                current = {path: self._file_signature(path) for path in affected_paths}
                if any(current[path] != previous_signatures[path]
                       for path in affected_paths):
                    # A different worker replaced the ROI while this worker waited.
                    return
            elif all(path.is_file() for path in affected_paths):
                return

            print(
                f'[AllClearAEF] {reason}: {len(affected_paths)} raster(s) for '
                f'{roi}; automatically re-downloading the ROI.',
                flush=True,
            )
            for path in affected_paths:
                print(f'[AllClearAEF] Affected: {path}', flush=True)
            command = [
                sys.executable,
                str(self.downloader_path),
                '--roi', roi,
                '--force',
                '--cpus', '1',
                '--output-dir', str(self.data_root),
            ]
            result = subprocess.run(
                command,
                cwd=str(self.root),
                text=True,
                capture_output=True,
                timeout=self.download_timeout,
                check=False,
            )
            if result.stdout:
                print(result.stdout.rstrip(), flush=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f'Automatic download failed for {roi} (exit {result.returncode}).\n'
                    f'{result.stderr.rstrip()}')

            remaining = [path for path in affected_paths if not path.is_file()]
            if remaining:
                missing = '\n'.join(str(path) for path in remaining)
                raise FileNotFoundError(
                    f'ROI {roi} was re-downloaded, but required raster(s) are '
                    f'still missing:\n{missing}')

    @staticmethod
    def _file_signature(path: Path):
        try:
            stat = path.stat()
            return stat.st_size, stat.st_mtime_ns
        except FileNotFoundError:
            return None

    def _download_missing_roi(self, sample, missing_paths):
        self._run_roi_download(sample, missing_paths, reason='Missing AllClear data')

    def _repair_corrupt_roi(self, sample, path, error):
        signature = {path: self._file_signature(path)}
        print(
            f'[AllClearAEF] Rasterio could not read {path}: {error}',
            flush=True,
        )
        self._run_roi_download(
            sample,
            [path],
            reason='Corrupt/unreadable AllClear data',
            previous_signatures=signature,
        )

    def _ensure_allclear_files(self, sample):
        paths = [sample['cloudy'], sample['target']]
        if self.load_sar:
            paths.insert(0, sample['SAR'])
        if self.include_cloud_mask:
            paths.append(sample['cloud_mask'])
        missing = [path for path in paths if not path.is_file()]
        if missing:
            self._download_missing_roi(sample, missing)

    @staticmethod
    def _shape(path: Path) -> tuple[int, int]:
        try:
            with rasterio.open(path) as src:
                return src.height, src.width
        except rasterio.errors.RasterioIOError as error:
            raise _RasterAccessError(path, error) from error

    def _window(self, sample, idx: int) -> Window | None:
        if self.output_size is None and self.crop_size is None:
            return None
        keys = ['cloudy', 'target']
        if self.load_sar:
            keys.insert(0, 'SAR')
        if self.load_aef:
            keys.insert(0, 'AEF')
        if self.include_cloud_mask:
            keys.append('cloud_mask')
        shapes = [self._shape(sample[key]) for key in keys]
        if len(set(shapes)) != 1:
            raise ValueError(f'Raster shape mismatch for {sample["sample_id"]}: {shapes}')
        height, width = shapes[0]
        # Training uses the configured random crop (128x128 in the paper),
        # while validation omits crop_size and keeps the full output window
        # (256x256 for AllClear).
        size = self.crop_size if self.crop_size is not None else self.output_size
        if height < size or width < size:
            raise ValueError(
                f'Raster {height}x{width} is smaller than requested {size}x{size} '
                f'for {sample["sample_id"]}')
        if self.crop_size is not None:
            # Deterministic per index/worker call; Lightning seeds NumPy workers.
            top = np.random.randint(0, height - size + 1)
            left = np.random.randint(0, width - size + 1)
        else:
            top = (height - size) // 2
            left = (width - size) // 2
        return Window(left, top, size, size)

    @staticmethod
    def _read(path: Path, window: Window | None) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f'Raster not found: {path}')
        try:
            with rasterio.open(path) as src:
                return src.read(window=window, out_dtype='float32')
        except rasterio.errors.RasterioIOError as error:
            raise _RasterAccessError(path, error) from error

    def _load_arrays(self, sample, idx):
        window = self._window(sample, idx)
        arrays = []
        if self.load_aef:
            arrays.append(self._read(sample['AEF'], window))
        if self.load_sar:
            arrays.append(self._read(sample['SAR'], window))
        arrays.extend((
            self._read(sample['cloudy'], window),
            self._read(sample['target'], window),
        ))
        if self.include_cloud_mask:
            arrays.append(self._read(sample['cloud_mask'], window))
        return tuple(arrays)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        self._ensure_allclear_files(sample)
        try:
            arrays = self._load_arrays(sample, idx)
        except _RasterAccessError as error:
            repairable = {sample['cloudy'], sample['target']}
            if self.load_sar:
                repairable.add(sample['SAR'])
            if self.include_cloud_mask:
                repairable.add(sample['cloud_mask'])
            if error.path not in repairable:
                raise rasterio.errors.RasterioIOError(
                    f'AEF raster is corrupt or unreadable and cannot be repaired '
                    f'by the AllClear downloader: {error.path}') from error.cause
            self._repair_corrupt_roi(sample, error.path, error.cause)
            try:
                # Recompute the window because the replacement raster was reopened.
                arrays = self._load_arrays(sample, idx)
            except _RasterAccessError as retry_error:
                raise rasterio.errors.RasterioIOError(
                    f'Raster remained unreadable after one automatic ROI repair: '
                    f'{retry_error.path}') from retry_error.cause

        cursor = 0
        aef = arrays[cursor] if self.load_aef else None
        cursor += int(self.load_aef)
        sar = arrays[cursor] if self.load_sar else None
        cursor += int(self.load_sar)
        cloudy, target = arrays[cursor:cursor + 2]
        cursor += 2
        cloud_mask = arrays[cursor] if self.include_cloud_mask else None

        if aef is not None and aef.shape[0] != 64:
            raise ValueError(f'Expected 64 AEF bands, got {aef.shape[0]}: {sample["AEF"]}')
        if sar is not None and sar.shape[0] != 2:
            raise ValueError(f'Expected 2 SAR bands, got {sar.shape[0]}: {sample["SAR"]}')
        if cloudy.shape[0] != 13 or target.shape[0] != 13:
            raise ValueError(
                f'Expected 13 S2 bands, got cloudy={cloudy.shape[0]}, '
                f'target={target.shape[0]} for {sample["sample_id"]}')

        if aef is not None:
            np.nan_to_num(aef, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if sar is not None:
            for band, band_min in zip(sar, self.sar_db_mins[:, 0, 0]):
                np.nan_to_num(
                    band, copy=False, nan=float(band_min),
                    posinf=self.sar_max, neginf=float(band_min))
        np.nan_to_num(cloudy, copy=False)
        np.nan_to_num(target, copy=False)
        # AlphaEarth pixels are unit-length 64-D embedding vectors. Preserve
        # their spherical geometry instead of clipping individual components.
        if aef is not None:
            aef_norm = np.linalg.norm(aef, axis=0, keepdims=True)
            aef = aef / np.maximum(aef_norm, np.float32(1e-6))
            if self.zero_aef:
                aef.fill(0.0)
        if sar is not None:
            # Paper preprocessing: band order is AllClear's VV, VH; clip the
            # two dB channels independently and rescale each to [0, 1].
            sar = np.clip(sar, self.sar_db_mins, self.sar_max)
            sar = (sar - self.sar_db_mins) / (self.sar_max - self.sar_db_mins)
        cloudy = np.clip(cloudy, 0.0, self.s2_max) / self.s2_max
        target = np.clip(target, 0.0, self.s2_max) / self.s2_max

        scale = self.data_range
        result = {
            'cloudy': (cloudy * scale).astype(np.float32, copy=False),
            'target': (target * scale).astype(np.float32, copy=False),
        }
        if aef is not None:
            result['AEF'] = (aef * scale).astype(np.float32, copy=False)
        if sar is not None:
            result['SAR'] = (sar * scale).astype(np.float32, copy=False)
        if cloud_mask is not None:
            if cloud_mask.shape[0] < 2:
                raise ValueError(
                    f'Expected at least 2 cld_shdw bands, got {cloud_mask.shape[0]}: '
                    f'{sample["cloud_mask"]}')
            # AllClear cld_shdw band 2 is the binary cloud mask.
            result['cloud_mask'] = (cloud_mask[1:2] > 0.5).astype(np.float32)
        return result
