"""Small offline checks for release configuration, gradients, and metric matching."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from config import find_config_using_name
from scripts.summarize_metrics import METRICS, summarize
from train import CloudRemovalModel


class ReleaseConfigTest(unittest.TestCase):
    def test_paper_defaults(self):
        for name in ('ecrformer_aef', 'ecrformer_s2only',
                     'ecrformer_aef_shallow_fusion', 'ecrformer_aef_zero_legacy',
                     'ecrformer_sar_legacy'):
            with self.subTest(config=name):
                config = find_config_using_name(name)()
                self.assertEqual(config.seed, 42)
                self.assertEqual(config.dataset.split, ['val'])
                self.assertEqual(config.dataset.train_ratio, 0.8)
                self.assertTrue(config.dataset.kwargs['preserve_manifest_roi_split'])
                self.assertFalse(config.dataset.kwargs['auto_download_missing'])
                self.assertEqual(config.train.train_bs * config.optim.accumulate_grad_batches, 16)

    def test_core_architecture_and_cpu_gradient(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        config = find_config_using_name('ecrformer_aef')()
        with contextlib.redirect_stdout(io.StringIO()):
            model = CloudRemovalModel(config)
            self.assertEqual(len(model.net.encoder), 3)
            self.assertEqual(len(model.net.decoder), 3)
            self.assertEqual(model.net.aef_adapter.project.out_channels, 16)
            self.assertEqual(torch.count_nonzero(model.net.aef_structure_gates), 0)
            self.assertEqual(torch.count_nonzero(model.net.aef_semantic_gate), 0)
            aef = torch.nn.functional.normalize(torch.randn(1, 64, 32, 32), dim=1)
            optical = torch.rand(1, 13, 32, 32)
            target = torch.rand_like(optical)
            model.eval()
            with torch.no_grad():
                prediction = model((aef, optical))[0]
                control = model((torch.zeros_like(aef), optical))[0]
            self.assertEqual(tuple(prediction.shape), (1, 13, 32, 32))
            torch.testing.assert_close(prediction, control, rtol=0, atol=0)
            model.train()
            prediction, _ = model((aef, optical))
            mae, structural = model._fp32_losses(prediction, target)
            loss = 0.9 * mae + 0.1 * structural
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.net.aef_structure_gates.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))


class MetricsSummaryTest(unittest.TestCase):
    def test_complete_common_selection_and_missing_id_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'split.json'
            snapshot.write_text(json.dumps({'partitions': {
                'validation': {'sample_ids': ['a', 'b'], 'roi_ids': ['roi1', 'roi2']},
                'sar_validation': {'sample_ids': ['b'], 'roi_ids': ['roi2']},
            }}))
            path = root / 'metrics.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['sample_id', 'roi', *METRICS])
                writer.writeheader()
                for sid, roi, value in [('a', 'roi1', 1), ('b', 'roi2', 2)]:
                    writer.writerow({'sample_id': sid, 'roi': roi,
                                     **{metric: value for metric in METRICS}})
            result = summarize(path, 'sar_validation', snapshot)
            self.assertEqual(result['num_samples'], 1)
            self.assertEqual(result['metrics']['PSNR'], 2)
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['sample_id', 'roi', *METRICS])
                writer.writeheader()
            with self.assertRaises(ValueError):
                summarize(path, 'validation', snapshot)


if __name__ == '__main__':
    unittest.main()
