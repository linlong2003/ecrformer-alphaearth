import torch
import torch.nn.functional as F
from torch import Tensor, nn
from timm.layers import to_2tuple, trunc_normal_
from einops import rearrange
from typing import Optional, Sequence

try:
    from models.module_util import LayerNorm, split_integer
    from models.module import (
        ChannelAttention, SpatialAttention, LinearAttnBlock,
        RestormerBlock, ECRformerBlock,
        TopkAttnBlock, UNetBottleneck,
    )
except ImportError:
    from module_util import LayerNorm, split_integer
    from module import (
        ChannelAttention, SpatialAttention, LinearAttnBlock,
        RestormerBlock, ECRformerBlock,
        TopkAttnBlock, UNetBottleneck,
    )

if __name__ != "__main__":
    print = lambda *args, **kwargs: ...


def format_shape(shape):
    if isinstance(shape, (list, tuple)):
        return f"({', '.join(str(s) for s in shape)})"
    return str(shape)


class ECRformerModel(nn.Module):
    def __init__(
        self,
        in_chans: int | Sequence[int] = 3,
        out_chans: int = 3,
        num_layers: int = 4,
        num_blocks: Sequence[int] = [4, 3, 2, 1],
        features_start: int = 64,
        drop_path_rate: float = 0.,
        bilinear: bool = False,
        cbam: Optional[str] = None,
        block_type: str | Sequence[str] = 'multi_dilated',
        conv_type: str = 'conv',
        norm_type: str = 'batch',
        decoupled_input: bool = True,
        bottle_neck: Optional[str] = 'topk',
        num_refine: Optional[int] = None,
        pos_encoding: Optional[str] = None,
        aef_conditioning: bool = False,
        aef_project_channels: int = 16,
    ) -> None:
        if num_layers < 1:
            raise ValueError(
                f"num_layers = {num_layers}, expected: num_layers > 0")

        assert len(num_blocks) == num_layers
        if isinstance(in_chans, int):
            in_chans = [in_chans]
        else:
            in_chans = list(in_chans)
        num_inputs = len(in_chans)

        dpr_cnt = 0
        dprs = [x.item() for x in torch.linspace(
            0, drop_path_rate, num_layers * 2 - 2)]

        down_block_type, up_block_type = to_2tuple(block_type)

        assert bottle_neck in ['topk', 'sa', 'tsa', None]
        assert pos_encoding in ['rand', 'sin', None]

        super().__init__()
        self.num_layers = num_layers
        self.decoupled_input = decoupled_input
        self.bottle_neck = bottle_neck
        self.in_chans = in_chans
        self.aef_conditioning = bool(aef_conditioning)
        self.expects_separate_inputs = self.aef_conditioning

        num_bottle_neck = num_blocks[-1]
        self.num_bottle_neck = num_bottle_neck

        conv_type = get_conv(conv_type)
        norm_type = get_norm(norm_type)

        # Stem. AEF conditioning keeps the two modalities separate: cloudy S2
        # forms the primary visual stream, while AEF is adapted and injected as
        # a controlled prior at multiple depths.
        if self.aef_conditioning:
            if not self.decoupled_input or len(in_chans) != 2:
                raise ValueError(
                    'aef_conditioning expects decoupled in_chans=[AEF, cloudy]')
            self.aef_channels, self.primary_channels = in_chans
            self.stem = nn.Conv2d(
                self.primary_channels, features_start, kernel_size=7, padding=3,
                padding_mode='reflect')
            self.aef_adapter = AEFTaskAdapter(
                self.aef_channels, aef_project_channels)
        elif self.decoupled_input:
            self.stem = DecoupledEncoder(
                in_ch_list=in_chans,
                out_ch_list=split_integer(features_start, num_inputs, [1, 1]),
                kernel_size=7, dilation=[1, ],
                conv_type=conv_type, norm_type=norm_type)
        else:
            if len(in_chans) != 1:
                raise ValueError(
                    f'Non-decoupled input expects one channel count, got {in_chans}')
            self.stem = nn.Conv2d(
                in_chans[0], features_start,
                kernel_size=7, padding=3, padding_mode='reflect')

        encoder_feats = [features_start * (2 ** i) for i in range(num_layers)]
        if self.aef_conditioning:
            self.aef_structure_proj = nn.ModuleList([
                nn.Conv2d(aef_project_channels, channels, kernel_size=1, bias=False)
                for channels in encoder_feats[:-1]
            ])
            self.aef_structure_gates = nn.Parameter(
                torch.zeros(len(encoder_feats) - 1))
            self.aef_semantic_adapter = nn.Sequential(
                nn.Conv2d(aef_project_channels, aef_project_channels, kernel_size=3,
                          padding=1, padding_mode='reflect', bias=False),
                nn.GroupNorm(4, aef_project_channels),
                nn.GELU(),
                nn.Conv2d(aef_project_channels, encoder_feats[-1],
                          kernel_size=1, bias=False),
            )
            self.aef_semantic_gate = nn.Parameter(torch.zeros(1))
        decoder_feats = encoder_feats[::-1]
        decoder_feats[-1] *= 2
        print(f"encoder_feats: {encoder_feats}")
        print(f"decoder_feats: {decoder_feats}")

        # Encoder
        encoder = []
        downsampler = []
        down_proj = []
        for idx in range(num_layers - 1):
            feats = encoder_feats[idx]
            next_feats = encoder_feats[idx + 1]

            down_post = []
            if isinstance(cbam, str):
                for item in cbam.lower().split('+'):
                    if '1ca' in item:
                        down_post.append(ChannelAttention(feats))
                    if '1sa' in item:
                        down_post.append(SpatialAttention())
                    if '1la' in item:
                        down_post.append(LinearAttnBlock(feats))
                    if '1ta' in item:
                        down_post.append(RestormerBlock(feats))

            down_fn = [get_block(down_block_type)(
                feats, drop_path_rate=dprs[dpr_cnt], conv_type=conv_type, norm_type=norm_type,
            ) for _ in range(num_blocks[idx])]

            dpr_cnt += 1
            encoder.append(nn.Sequential(*down_fn, *down_post))

            down_proj.append(nn.Conv2d(feats, out_chans,
                                       kernel_size=3, padding=1, padding_mode='reflect'))

            downsampler.append(nn.Sequential(
                nn.Conv2d(feats, next_feats // 4, kernel_size=3, stride=1,
                          padding=1, padding_mode='reflect', bias=False),
                nn.PixelUnshuffle(2),
                ChannelAttention(next_feats),
                SpatialAttention(),
            ))

        self.encoder = nn.ModuleList(encoder)
        self.downsampler = nn.ModuleList(downsampler)

        # Bottleneck
        feats = encoder_feats[-1]
        self.bottle_neck_module = None
        if bottle_neck and num_bottle_neck:
            if bottle_neck == 'sa':
                self.bottle_neck_module = nn.Sequential(
                    *[UNetBottleneck(feats, feats)
                      for _ in range(num_bottle_neck)]
                )
            elif bottle_neck == 'topk':
                self.bottle_neck_module = nn.Sequential(
                    *[TopkAttnBlock(feats) for _ in range(num_bottle_neck)],
                )
            elif bottle_neck == 'tsa':
                self.bottle_neck_module = nn.Sequential(
                    *[RestormerBlock(feats)
                      for _ in range(num_bottle_neck)],
                )

        neck_size = 256 // 2 ** (num_layers - 1)
        if pos_encoding == 'rand':
            self.learned_pos_embed = nn.Parameter(
                torch.zeros(1, feats, *to_2tuple(neck_size)))
            trunc_normal_(self.learned_pos_embed, std=.02)
        elif pos_encoding == 'sin':
            self.learned_pos_embed = nn.Parameter(posemb_sincos_2d(
                *to_2tuple(neck_size), feats) / 50)
        elif not pos_encoding:
            self.learned_pos_embed = 0.

        # Decoder
        decoder = []
        upsampler = []
        up_proj = []
        for idx in range(1, num_layers):
            last_feats = decoder_feats[idx - 1]
            feats = decoder_feats[idx]

            upsampler.append(
                nn.Sequential(
                    nn.Conv2d(last_feats, 4 * feats, kernel_size=3, stride=1,
                              padding=1, padding_mode='reflect', bias=False),
                    nn.PixelShuffle(2),
                    ChannelAttention(feats),
                    SpatialAttention())
                if last_feats != feats
                else nn.Sequential(
                    nn.Conv2d(last_feats, 2 * last_feats, kernel_size=3, stride=1,
                              padding=1, padding_mode='reflect', bias=False),
                    nn.PixelShuffle(2),
                    ChannelAttention(last_feats // 2),
                    SpatialAttention())
            )

            up_post = []
            if isinstance(cbam, str):
                for item in cbam.lower().split('+'):
                    if 'ca2' in item:
                        up_post.append(ChannelAttention(feats))
                    if 'sa2' in item:
                        up_post.append(SpatialAttention())
                    if 'la2' in item:
                        up_post.append(LinearAttnBlock(feats))
                    if 'ta2' in item:
                        up_post.append(RestormerBlock(feats))

            up_fn = [get_block(up_block_type)(
                feats, drop_path_rate=dprs[dpr_cnt], conv_type=conv_type, norm_type=norm_type)
                for _ in range(num_blocks[num_layers - idx - 1])]
            dpr_cnt += 1

            skip_feats = encoder_feats[-1 - idx]
            fuser = nn.Conv2d(feats + skip_feats, feats, kernel_size=3, padding=1,
                              padding_mode='reflect') if last_feats != feats else nn.Identity()
            decoder.append(nn.Sequential(fuser, *up_fn, *up_post))
            up_proj.append(nn.Conv2d(feats, out_chans,
                                     kernel_size=3, padding=1, padding_mode='reflect'))

        self.upsampler = nn.ModuleList(upsampler)
        self.decoder = nn.ModuleList(decoder)

        # Refine
        feats = decoder_feats[-1]
        self.refine = None
        if num_refine:
            refine = [RestormerBlock(feats, kernel_size=3, dilation=[1, 2, 4],
                                     conv_type=conv_type, norm_type=norm_type)
                      for _ in range(num_refine)]
            self.refine = nn.Sequential(*refine)

        # Final Convolution
        self.final_conv = nn.Conv2d(feats, out_chans,
                                    kernel_size=3, padding=1, padding_mode='reflect')

        self.down_proj = nn.ModuleList(down_proj)
        self.up_proj = nn.ModuleList(up_proj)

    def forward(self, x: Tensor | Sequence[Tensor], return_map=False) -> Tensor:
        # Stem
        if self.aef_conditioning:
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise TypeError(
                    'AEF-conditioned ECRformer expects (aef, cloudy) tensors.')
            aef, primary = x
            if aef.shape[1] != self.aef_channels:
                raise ValueError(
                    f'Expected {self.aef_channels} AEF channels, got {aef.shape[1]}')
            if primary.shape[1] != self.primary_channels:
                raise ValueError(
                    f'Expected {self.primary_channels} primary channels, got '
                    f'{primary.shape[1]}')
            aef_features = self.aef_adapter(aef)
            x_temp = self.stem(primary)
            input_shape = [tuple(aef.shape), tuple(primary.shape)]
        else:
            aef_features = None
            x_temp = self.stem(x)
            input_shape = x.shape
        x_enc = [x_temp]
        print(f"Stem: {format_shape(input_shape)} -> {format_shape(x_temp.shape)}")

        # Encoder path
        print("\nEncoder path:")
        down_projs = []
        for i in range(len(self.encoder)):
            down = self.downsampler[i]
            layer = self.encoder[i]
            proj = self.down_proj[i]

            if self.aef_conditioning:
                aef_scale = F.adaptive_avg_pool2d(
                    aef_features, x_temp.shape[-2:])
                aef_delta = self.aef_structure_proj[i](aef_scale)
                x_temp = x_temp + torch.tanh(self.aef_structure_gates[i]) * aef_delta

            x_out = layer(x_temp)
            x_enc.append(x_out)
            x_down = down(x_out)

            print(f"Input: {format_shape(x_temp.shape)}, "
                  f"Encoder_{i} out: {format_shape(x_out.shape)}, "
                  f"Down: {format_shape(x_down.shape)}")

            mid = proj(x_out)
            print(f"Down proj_{i}: {format_shape(x_enc[i+1].shape)}"
                  f" -> {format_shape(mid.shape)}")
            down_projs.append(mid)

            x_temp = x_down

        # Bottleneck
        print("\nBottleneck:")
        neck = x_temp
        if self.aef_conditioning:
            aef_neck = F.adaptive_avg_pool2d(
                aef_features, neck.shape[-2:])
            semantic_delta = self.aef_semantic_adapter(aef_neck)
            neck = neck + torch.tanh(self.aef_semantic_gate) * semantic_delta
        if self.bottle_neck_module and self.num_bottle_neck:
            neck = self.bottle_neck_module(neck + self.learned_pos_embed)
            print(f"Bottleneck: {neck.shape}")

        # Decoder path
        print("\nDecoder path:")
        x_temp = neck
        up_projs = []
        x_dec = []

        for i in range(len(self.decoder)):
            up = self.upsampler[i]
            layer = self.decoder[i]
            proj = self.up_proj[i]

            skip = x_enc[-1 - i]
            x_up = up(x_temp)

            print(f"Up: {format_shape(x_temp.shape)} -> "
                  f"{format_shape(x_up.shape)}, "
                  f"skip: {format_shape(skip.shape)}")

            x_fuse = torch.cat([x_up, skip], dim=1)
            x_out = layer(x_fuse)
            x_dec.append(x_out)

            print(f"Fuse (x_up+skip): {format_shape(x_fuse.shape)}, "
                  f"Decoder_{i} out: {format_shape(x_out.shape)}")

            mid = proj(x_out)
            print(f"Up proj_{i}: {format_shape(x_dec[i].shape)}"
                  f" -> {format_shape(mid.shape)}")
            up_projs.append(mid)

            x_temp = x_out

        # Refine
        if self.refine:
            print("\nRefine:")
            x_temp = self.refine(x_temp)
            print(f"Refine out: {format_shape(x_temp.shape)}")

        x_dec.append(x_temp)

        # Final Convolution
        x_temp = self.final_conv(x_temp)
        print(f'Final out: {format_shape(x_temp.shape)}')

        if not return_map:
            return x_temp, (down_projs, up_projs)
        else:
            return x_temp, (down_projs, up_projs), x_enc, x_dec


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def get_conv(conv_type):
    if conv_type == 'conv':
        return nn.Conv2d
    else:
        raise ValueError(f"Unsupported conv_type: {conv_type}")


def get_norm(norm_type):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'layer':
        return LayerNorm
    elif norm_type == 'group':
        return nn.GroupNorm
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


def get_block(block_type):
    block_type = block_type.lower().replace('_', '')
    if block_type == 'restormer':
        return RestormerBlock
    elif block_type == 'ecrformer':
        return ECRformerBlock
    else:
        raise ValueError(f"Invalid block type: {block_type}")


class AEFTaskAdapter(nn.Module):
    """Learn a compact, task-aligned per-pixel AEF representation."""

    def __init__(self, in_chans: int = 64, project_chans: int = 16):
        super().__init__()
        if project_chans <= 0 or project_chans % 4 != 0:
            raise ValueError(
                f'project_chans must be a positive multiple of 4, got {project_chans}')
        self.project = nn.Conv2d(
            in_chans, project_chans, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(4, project_chans)
        self.act = nn.GELU()
        self.spatial = nn.Conv2d(
            project_chans, project_chans, kernel_size=3, padding=1,
            groups=project_chans, padding_mode='reflect', bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(self.act(self.norm(self.project(x))))


class DecoupledEncoder(nn.Module):
    """Decoupled input encoder for separate SAR/optical branches."""

    def __init__(self, in_ch_list: list[int], out_ch_list: list[int],
                 kernel_size: int = 7, dilation: list[int] = [1, ],
                 conv_type=nn.Conv2d, norm_type=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.in_ch_list = in_ch_list
        branch = []
        for in_ch, out_ch in zip(in_ch_list, out_ch_list):
            print(f"DecoupledEncoder: {in_ch} -> {out_ch}")
            branch.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                    padding=kernel_size // 2, padding_mode='reflect'))
        self.branch = nn.ModuleList(branch)

    def forward(self, x):
        inputs = torch.split(x, self.in_ch_list, dim=1)
        outputs = [f(x) for f, x in zip(self.branch, inputs)]
        return torch.cat(outputs, dim=1)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    pe = rearrange(pe, '(h w) c -> 1 c h w', h=h, w=w)
    return pe.type(dtype)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ECRformer
    net = ECRformerModel(
        in_chans=[2, 13], out_chans=13, cbam='1ca2+1sa2', pos_encoding=None,
        num_layers=4, num_blocks=[2, 3, 2, 2], bottle_neck='tsa', num_refine=4,
        features_start=48, drop_path_rate=0., conv_type='conv', norm_type='batch',
        block_type=['ecrformer', 'ecrformer'], decoupled_input=True)
    # ECRformer-Light
    net = ECRformerModel(
        in_chans=[2, 13], out_chans=13, cbam='1ca2+1sa2', pos_encoding=None,
        num_layers=4, num_blocks=[2, 2, 1, 1], bottle_neck='tsa', num_refine=2,
        features_start=32, drop_path_rate=0., conv_type='conv', norm_type='batch',
        block_type=['ecrformer', 'ecrformer'], decoupled_input=True)

    def count_parameters(model, mode='return'):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def smart_print(num, title=''):
        units = ['', 'K', 'M', 'G']
        unit_idx = 0
        while num > 1000:
            num /= 1000
            unit_idx += 1
        prefix = f"{title}: " if title else ""
        print(f"{prefix}{num:.4f}{units[unit_idx]}")

    total = count_parameters(net)
    down_proj_params = count_parameters(net.down_proj)
    up_proj_params = count_parameters(net.up_proj)
    smart_print(total - down_proj_params - up_proj_params)
    smart_print(down_proj_params)
    smart_print(up_proj_params)
