import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser, Namespace

from util.pytorch_ssim import SSIM
from util.util import count_parameters, initialize_weights, compute_metric
from util.augment import TestAugment, TrainAugment
from util.checkpoint import find_latest_checkpoint
from util.data_split import build_train_valid_datasets

from models import find_model_using_name
from data import find_dataset_using_name
from config import find_config_using_name


class CloudRemovalModel(pl.LightningModule):
    """PyTorch Lightning module for cloud removal with ECRformer.

    Supports multi-scale projection loss (down_proj + up_proj).
    """

    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.input = config.net.input
        self.output = config.net.output

        net_class = find_model_using_name(config.net.name)
        self.net = net_class(**config.net.cfg)

        self.SSIM = SSIM()
        self.downsample = nn.AvgPool2d(2, 2)

        self.loss_weight = list(config.train.loss_weight[:2])
        if len(self.loss_weight) != 2:
            raise ValueError(
                f'loss_weight must provide at least two entries, got {len(config.train.loss_weight)}'
            )

        self.proj_weight = config.train.proj_weight

        self.lr = config.train.lr
        self.compute_lpips = getattr(config.train, 'compute_lpips', True)
        self.train_augment = TrainAugment(crop_size=config.dataset.crop_size)
        self.save_hyperparameters()

        initialize_weights(self.net)

        down_weight, up_weight = self.proj_weight
        if down_weight == 0.:
            self.net.down_proj.requires_grad_(False)
        if up_weight == 0.:
            self.net.up_proj.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    @staticmethod
    def _assert_finite(name, tensor, batch_idx=None):
        if isinstance(tensor, (list, tuple)):
            for index, value in enumerate(tensor):
                CloudRemovalModel._assert_finite(
                    f'{name}[{index}]', value, batch_idx)
            return
        if isinstance(tensor, dict):
            for key, value in tensor.items():
                CloudRemovalModel._assert_finite(
                    f'{name}[{key}]', value, batch_idx)
            return
        if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
            location = f' at batch {batch_idx}' if batch_idx is not None else ''
            raise FloatingPointError(
                f'NaN/Inf detected in {name}{location}; training stopped immediately.')

    def _fp32_losses(self, pred, target):
        """Compute differentiable L1 and SSIM losses outside AMP autocast."""
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            mae = nn.functional.l1_loss(pred_fp32, target_fp32)
            ssim_loss = 1.0 - self.SSIM(pred_fp32, target_fp32)
        return mae, ssim_loss

    def _fp32_sdfl_losses(self, encoder_pred, decoder_pred, target):
        """Compute the paper-original encoder SSIM and decoder L1 terms in FP32."""
        with torch.autocast(device_type=target.device.type, enabled=False):
            target_fp32 = target.float()
            encoder_loss = 1.0 - self.SSIM(encoder_pred.float(), target_fp32)
            decoder_loss = nn.functional.l1_loss(decoder_pred.float(), target_fp32)
        return encoder_loss, decoder_loss

    @torch.no_grad()
    def fuse_input(self, batch):
        """Prepare separate conditioned inputs (or legacy concatenated inputs)."""
        if self.training:
            batch = self.train_augment.augment(batch)
        target = torch.cat([batch[key] for key in self.output], dim=1)
        input_tensors = tuple(batch[key] for key in self.input)
        if getattr(self.net, 'expects_separate_inputs', False):
            model_input = input_tensors
        else:
            model_input = torch.cat(input_tensors, dim=1)
        return batch, model_input, target

    def training_step(self, batch, batch_idx):
        batch, model_input, target = self.fuse_input(batch)
        self._assert_finite('training input', model_input, batch_idx)
        self._assert_finite('training target', target, batch_idx)
        pred, projs = self.forward(model_input)
        self._assert_finite('model prediction', pred, batch_idx)

        down_projs, up_projs = projs

        # Main loss is always evaluated in FP32, even when the model runs in AMP.
        mae_loss, ssim_loss = self._fp32_losses(pred, target)
        self._assert_finite('MAE loss', mae_loss, batch_idx)
        self._assert_finite('SSIM loss', ssim_loss, batch_idx)
        loss = mae_loss * self.loss_weight[0] + ssim_loss * self.loss_weight[1]

        # Paper-original Semantic-Decoupled Feature Learning (SDFL):
        # encoder projections recover structure with (1 - SSIM), while decoder
        # projections recover texture with L1. All terms are evaluated in FP32.
        down_weight, up_weight = self.proj_weight
        if down_weight or up_weight:
            down_loss_sum = loss.new_zeros(())
            up_loss_sum = loss.new_zeros(())
            mid_target = target
            for down_proj, up_proj in zip(down_projs, up_projs[::-1]):
                encoder_loss, decoder_loss = self._fp32_sdfl_losses(
                    down_proj, up_proj, mid_target)
                down_loss_sum = down_loss_sum + encoder_loss
                up_loss_sum = up_loss_sum + decoder_loss
                mid_target = self.downsample(mid_target.float())
            down_loss = down_loss_sum / max(len(down_projs), 1)
            up_loss = up_loss_sum / max(len(up_projs), 1)
            self._assert_finite('down projection loss', down_loss, batch_idx)
            self._assert_finite('up projection loss', up_loss, batch_idx)
            loss = loss + down_loss * down_weight + up_loss * up_weight
            self.log('down_proj', down_loss, prog_bar=True)
            self.log('up_proj', up_loss, prog_bar=True)

        self._assert_finite('total loss', loss, batch_idx)
        self.log('train_MAE', mae_loss, prog_bar=True)
        self.log('train_SSIM', 1 - ssim_loss, prog_bar=True)
        if getattr(self.net, 'aef_conditioning', False):
            for index, gate in enumerate(self.net.aef_structure_gates):
                self.log(f'aef_structure_gate_{index}', torch.tanh(gate.detach()),
                         on_step=False, on_epoch=True)
            self.log('aef_semantic_gate',
                     torch.tanh(self.net.aef_semantic_gate.detach()),
                     on_step=False, on_epoch=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Stop on the first non-finite unscaled gradient, including under AMP."""
        for name, parameter in self.net.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f'NaN/Inf detected in unscaled gradient {name} at global step '
                    f'{self.global_step}; training stopped immediately.')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Also detect parameters corrupted by an optimizer update."""
        for name, parameter in self.net.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    f'NaN/Inf detected in parameter {name} after global step '
                    f'{self.global_step}; training stopped immediately.')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, model_input, target = self.fuse_input(batch)
        self._assert_finite('validation input', model_input, batch_idx)
        self._assert_finite('validation target', target, batch_idx)
        pred, projs = self.forward(model_input)
        self._assert_finite('validation prediction', pred, batch_idx)

        metrics = compute_metric(pred, target, size_average=True,
                                 include_lpips=self.compute_lpips)
        for name, value in metrics.items():
            self._assert_finite(f'validation {name}', value, batch_idx)
        self.log('valid_RMSE', metrics['RMSE'], prog_bar=False, on_epoch=True)
        self.log('valid_MAE', metrics['MAE'], prog_bar=True, on_epoch=True)
        self.log('valid_PSNR', metrics['PSNR'], prog_bar=False, on_epoch=True)
        self.log('valid_SAM', metrics['SAM'], prog_bar=False, on_epoch=True)
        self.log('valid_SSIM', metrics['SSIM'], prog_bar=True, on_epoch=True)
        if self.compute_lpips:
            self.log('valid_LPIPS', metrics['LPIPS'], prog_bar=False, on_epoch=True)
        valid_loss = metrics['MAE'] * self.loss_weight[0] + \
            (1 - metrics['SSIM']) * self.loss_weight[1]
        self._assert_finite('validation loss', valid_loss, batch_idx)
        self.log('valid_loss', valid_loss, prog_bar=False, on_epoch=True)
        if len(self.trainer.optimizers) > 0:
            self.log('learning_rate',
                     self.trainer.optimizers[0].param_groups[0]['lr'],
                     prog_bar=False, on_epoch=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        batch, model_input, target = self.fuse_input(batch)
        pred_result = self.forward(model_input)
        return batch, pred_result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',
                'interval': 'epoch',
                'frequency': 1,
            },
        }


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def main(config):
    torch.set_float32_matmul_precision("highest")
    pl.seed_everything(config.seed)

    print("\nBuilding model...")
    model = CloudRemovalModel(config)
    print(f"Model class: {model.net.__class__.__name__}")
    count_parameters(model)

    print("\nLoading dataset...")
    dataset_class = find_dataset_using_name(config.dataset.name)
    train_dataset, valid_dataset, split_info = build_train_valid_datasets(
        config, dataset_class)
    print(f"Training split: {split_info['train']}")
    print(f"Validation split: {split_info['valid']}")

    num_workers = config.train.num_workers
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.train_bs, drop_last=True,
        shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.train.valid_bs,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss', verbose=False, mode='min',
        auto_insert_metric_name=True, save_last=True, save_top_k=1)
    early_stop_callback = EarlyStopping(
        monitor='valid_loss', patience=config.train.early_stop,
        verbose=True, mode='min')

    log_name = config_name
    if config.name:
        log_name = f"{log_name}_{config.name}"

    save_dir = config.train.save_dir

    ckpt_path = config.train.ckpt_path
    resume_version = None

    if ckpt_path is None and not getattr(config, 'no_resume', False):
        auto_ckpt_path, version_num = find_latest_checkpoint(
            save_dir, log_name)
        if auto_ckpt_path is not None:
            ckpt_path = auto_ckpt_path
            resume_version = version_num
            print(f"Auto-resume checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Training from scratch.")
    elif getattr(config, 'no_resume', False):
        print("Auto-resume is disabled.")
        ckpt_path = None

    if resume_version is not None:
        tb_logger = TensorBoardLogger(
            save_dir=save_dir, name=log_name, version=resume_version)
    else:
        tb_logger = TensorBoardLogger(save_dir=save_dir, name=log_name)

    print("\nCreating trainer...")
    trainer = pl.Trainer(
        max_epochs=config.train.max_epoch,
        gradient_clip_val=.5,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[tb_logger],
        devices=config.train.gpu,
        **config.optim.__dict__,
    )

    print("\nStarting training...")
    print(f"Experiment directory: {os.path.join(save_dir, log_name)}")
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader, ckpt_path=ckpt_path)
    print("\nTraining finished.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='ecrformer_aef', help="Config name")
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--no-resume', action='store_true',
                        help="Disable automatic checkpoint resume")
    args = parser.parse_args()

    config_name = args.config
    print(f"Config: {config_name}")
    config_class = find_config_using_name(config_name)
    config = config_class()
    config.name = args.name
    config.train.gpu = [args.gpu]
    config.no_resume = args.no_resume

    main(config)
