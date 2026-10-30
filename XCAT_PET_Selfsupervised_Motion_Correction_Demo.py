
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    matrix_xyz: Tuple[int, int, int] = (144, 144, 355)
    spacing_xyz_mm: Tuple[float, float, float] = (2.0, 2.0, 1.9966)

    keep_z_start: int = 99
    keep_z_stop: int = 259

    n_phases: int = 40
    n_cardiac_gates: int = 8
    microframe_seconds: float = 0.125

    train_phases: Tuple[int, ...] = tuple(range(1, 33))
    val_phases: Tuple[int, ...] = tuple(range(33, 41))
    test_phases: Tuple[int, ...] = tuple(range(1, 41))

    # Data normalization. "percentile" is robust to PET outliers.
    normalization: str = "percentile"
    norm_low_percentile: float = 1.0
    norm_high_percentile: float = 99.5

    # Self-supervised loss
    w_photo: float = 1.0
    w_smooth: float = 0.05
    w_grad: float = 0.10
    w_mask_photo: float = 1.0
    smooth_edge_aware: bool = True

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 1
    num_workers: int = 2
    epochs: int = 100
    grad_clip: float = 1.0
    use_amp: bool = True
    seed: int = 2026

    # Model
    base_channels: int = 16
    attention_heads: int = 4

    # Training pair definition
    # "adjacent" means ph -> ph+1, never crossing train/val boundaries.
    pair_mode: str = "adjacent"

    # Inference / gate reference:
    # For each cardiac gate, use the first occurrence k as the reference.
    # Other occurrences k+8, k+16, k+24, k+32 are aligned to it.
    gate_reference_occurrence: int = 0

    # True XCAT GT:
    # Set to directory containing XCAT forward DVFs when available.
    xcat_gt_dir: Optional[str] = None
    xcat_gt_direction: str = "forward"  # forward = ref/source -> target/moving
    xcat_gt_format: str = "npy"          # npy, npz, binary
    xcat_gt_template: str = "p{phantom}_ph{phase:02d}.npy"
    xcat_gt_dtype: str = "float32"
    xcat_gt_shape_xyz: Tuple[int, int, int] = (144, 144, 355)
    xcat_gt_channels_first: bool = True

    # Raw binary storage convention. Header matrix remains (X,Y,Z), but the
    # binary stream is read as (Z,Y,X), i.e. z,y,x / D,H,W for PyTorch.
    raw_storage_order: str = "zyx"

    # Whole-volume displacement summaries should exclude zero/background
    # voxels. This threshold is applied to the reference PET image.
    valid_image_threshold: float = 1e-6

    # Optional conventional registration for qualitative MVF presentation.
    # It is deliberately kept outside training and quantitative GT evaluation.
    registration_enabled: bool = False
    registration_iterations: int = 80
    registration_shrink: int = 2
    registration_smoothing_sigma_mm: float = 1.0
    registration_update_sigma_mm: float = 0.5

    # Output / checkpointing
    save_every_epoch: bool = False  # legacy option; normally keep False
    resume: bool = True


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is useful for this controlled preliminary experiment.
    torch.backends.cudnn.benchmark = True


# ============================================================================
# Interfile / .i handling
# ============================================================================

def find_hdr_for_i(i_path: Path) -> Optional[Path]:
    candidates = [
        Path(str(i_path) + ".hdr"),
        i_path.with_suffix(".hdr"),
        i_path.parent / (i_path.name + ".hdr"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_interfile_header(hdr_path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {}
    text = hdr_path.read_text(errors="ignore")

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or ":=" not in line:
            continue
        key, value = [x.strip() for x in line.split(":=", 1)]
        key_l = key.lower()

        if key_l == "number format":
            info["number_format"] = value.lower()
        elif key_l == "number of bytes per pixel":
            try:
                info["bytes_per_pixel"] = int(float(value))
            except Exception:
                pass
        elif key_l.startswith("matrix size ["):
            m = re.search(r"\[(\d+)\]", key_l)
            if m:
                info[f"matrix_size_{m.group(1)}"] = int(float(value))
        elif key_l.startswith("scaling factor (mm/pixel) ["):
            m = re.search(r"\[(\d+)\]", key_l)
            if m:
                info[f"spacing_{m.group(1)}"] = float(value)

    # Expected current dataset is float32.
    bpp = info.get("bytes_per_pixel")
    fmt = str(info.get("number_format", ""))
    if bpp == 4 and "float" in fmt:
        info["dtype"] = np.dtype("<f4")
    elif bpp == 2 and "signed integer" in fmt:
        info["dtype"] = np.dtype("<i2")
    elif bpp == 2 and "unsigned integer" in fmt:
        info["dtype"] = np.dtype("<u2")
    elif bpp == 1:
        info["dtype"] = np.dtype("u1")
    else:
        info["dtype"] = None

    sizes = tuple(
        info.get(f"matrix_size_{i}") for i in (1, 2, 3)
    )
    if all(x is not None for x in sizes):
        info["matrix_xyz"] = tuple(int(x) for x in sizes)

    spacing = tuple(
        info.get(f"spacing_{i}") for i in (1, 2, 3)
    )
    if all(x is not None for x in spacing):
        info["spacing_xyz_mm"] = tuple(float(x) for x in spacing)

    return info


def read_i_dhw(
    i_path: Path,
    matrix_xyz: Tuple[int, int, int],
    expected_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Read the raw Interfile .i binary directly as (Z,Y,X) = (D,H,W).

    IMPORTANT:
        Interfile header:
            matrix size [1] = X = 144
            matrix size [2] = Y = 144
            matrix size [3] = Z = 355

        Raw binary storage used by this dataset is interpreted as:
            data.reshape(Z, Y, X)

    This is the convention requested for the XCAT PET files. The resulting
    array is already in PyTorch spatial order (D,H,W), so NO transpose is
    performed afterwards.
    """
    X, Y, Z = matrix_xyz
    n = X * Y * Z
    hdr = find_hdr_for_i(i_path)
    dtype = expected_dtype

    if dtype is None and hdr is not None:
        h = parse_interfile_header(hdr)
        dtype = h.get("dtype")

    if dtype is None:
        dtype = np.dtype("<f4")

    arr = np.fromfile(i_path, dtype=dtype, count=n)
    if arr.size != n:
        raise RuntimeError(
            f"{i_path}: read {arr.size} elements, expected {n}. "
            f"Check dtype, header, and matrix size."
        )

    # The raw stream is Z,Y,X. Header matrix sizes remain X,Y,Z.
    return arr.reshape((Z, Y, X)).astype(np.float32, copy=False)


# Backward-compatible name used nowhere internally after this revision.
# It intentionally returns D,H,W rather than pretending the raw bytes are XYZ.
def read_i_xyz(
    i_path: Path,
    matrix_xyz: Tuple[int, int, int],
    expected_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    return read_i_dhw(i_path, matrix_xyz, expected_dtype)


def write_float32_xyz_i(
    arr_xyz: np.ndarray,
    out_i: Path,
    spacing_xyz_mm: Tuple[float, float, float],
) -> None:
    """
    Save a float32 raw .i volume plus a minimal Interfile header.
    """
    out_i.parent.mkdir(parents=True, exist_ok=True)
    arr_xyz.astype("<f4", copy=False).tofile(out_i)

    X, Y, Z = arr_xyz.shape
    sx, sy, sz = spacing_xyz_mm
    hdr = out_i.with_name(out_i.name + ".hdr")
    hdr.write_text(
        "!INTERFILE\n"
        f"name of data file := {out_i.name}\n"
        "data format := image\n"
        "number format := float\n"
        "number of bytes per pixel := 4\n"
        "number of dimensions := 3\n"
        ";dimensions 1-3: x, y, z\n"
        f"matrix size [1] := {X}\n"
        f"matrix size [2] := {Y}\n"
        f"matrix size [3] := {Z}\n"
        f"scaling factor (mm/pixel) [1] := {sx:.6f}\n"
        f"scaling factor (mm/pixel) [2] := {sy:.6f}\n"
        f"scaling factor (mm/pixel) [3] := {sz:.6f}\n"
    )


# ============================================================================
# Heart mask generation
# ============================================================================

def generate_phase_masks_and_save_i(
    data_dir: str | Path,
    out_dir: str | Path,
    phases: Sequence[int] = tuple(range(1, 9)),
    frame: int = 36,
    matrix_xyz: Tuple[int, int, int] = (144, 144, 355),
    keep_z: slice = slice(99, 259),
    mask_threshold: float = 0.2,
) -> Dict[int, np.ndarray]:
    """
    Generate the phase masks using the user's established threshold logic.

    The raw PET stream is read as (Z,Y,X). Cropping is therefore performed on
    axis 0, and the returned/saved mask is directly (D,H,W).

    Threshold logic is preserved:
        threshold = 0.20 * max(activity)
        if empty -> mean(activity)
        if still empty -> activity > 0

    No mask is silently replaced by all ones.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks: Dict[int, np.ndarray] = {}
    for ph in phases:
        matches = sorted(data_dir.glob(f"*ph{ph}_fr{frame}.i"))
        if not matches:
            raise FileNotFoundError(
                f"Cannot generate mask for phase {ph}: no *ph{ph}_fr{frame}.i in {data_dir}"
            )

        vol_dhw = read_i_dhw(matches[0], matrix_xyz, expected_dtype=np.dtype("<f4"))
        vol_dhw = vol_dhw[keep_z, :, :].astype(np.float32, copy=False)

        finite = np.isfinite(vol_dhw)
        maxv = float(np.nanmax(vol_dhw)) if np.any(finite) else 0.0
        if not np.isfinite(maxv) or maxv <= 0:
            raise ValueError(f"Phase {ph} has no positive finite activity.")

        threshold = maxv * float(mask_threshold)
        mask = vol_dhw > threshold
        used_fallback = None

        if not np.any(mask):
            threshold = float(np.nanmean(vol_dhw))
            mask = vol_dhw > threshold
            used_fallback = "mean"

        if not np.any(mask):
            mask = vol_dhw > 0.0
            used_fallback = "gt0" if used_fallback is None else used_fallback + "+gt0"

        if not np.any(mask):
            raise ValueError(f"Mask for phase {ph} is empty after all fallbacks.")

        mask_dhw = mask.astype(np.uint8)
        np.save(out_dir / f"ph{ph:02d}_fr{frame}_heart_mask_dhw.npy", mask_dhw)
        masks[ph] = mask_dhw.astype(np.float32)

        print(
            f"Mask {data_dir.name} ph{ph:02d}: shape={mask_dhw.shape}, "
            f"voxels={int(mask_dhw.sum()):,}, fallback={used_fallback}"
        )

    np.save(
        out_dir / f"masks_ph{min(phases):02d}_to_ph{max(phases):02d}_dhw.npy",
        np.stack([masks[p] for p in phases], axis=0).astype(np.uint8),
    )
    return masks


def load_phase_masks(
    mask_dir: str | Path,
    phases: Sequence[int] = tuple(range(1, 9)),
    expected_shape_dhw: Tuple[int, int, int] = (160, 144, 144),
) -> Dict[int, np.ndarray]:
    """Load regenerated phase-specific masks in (D,H,W) order."""
    mask_dir = Path(mask_dir)
    masks: Dict[int, np.ndarray] = {}

    for ph in phases:
        candidates = [
            mask_dir / f"ph{ph:02d}_fr36_heart_mask_dhw.npy",
            mask_dir / f"p80_ph{ph:02d}_fr36_heart_mask_dhw.npy",
            mask_dir / f"phase_{ph}" / f"phase_ph{ph}_fr36_mask_xyz_crop.npy",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No regenerated mask found for phase {ph} in {mask_dir}. "
                f"Run --generate_masks first."
            )

        m = np.load(path)
        if m.shape != expected_shape_dhw:
            # Legacy mask may be stored as X,Y,Z crop. Recognize only the
            # exact expected permutation; do not guess arbitrary shapes.
            if m.shape == (expected_shape_dhw[2], expected_shape_dhw[1], expected_shape_dhw[0]):
                m = np.transpose(m, (2, 1, 0))
            else:
                raise ValueError(
                    f"Mask {path} has shape {m.shape}; expected {expected_shape_dhw}."
                )
        masks[ph] = (m > 0).astype(np.float32)

    return masks


def mask_for_phase(
    phase: int,
    phase_masks: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    The available masks are phase 1..8. Every temporal state maps to its
    cardiac phase by modulo-8:
        1,9,17,25,33 -> mask 1
        ...
        8,16,24,32,40 -> mask 8
    """
    cardiac_phase = ((phase - 1) % 8) + 1
    return phase_masks[cardiac_phase]


# ============================================================================
# Dataset
# ============================================================================

PHASE_RE = re.compile(r"ph(\d+)_fr(\d+)", re.IGNORECASE)


class XCATPETDataset(Dataset):
    """
    Explicitly loads one selected fr-level (default fr36).

    __getitem__ returns:
        reference_image: (1,D,H,W)
        moving_image:    (1,D,H,W)
        reference_mask:  (1,D,H,W)
        moving_mask:     (1,D,H,W)
        metadata dict

    Pair direction:
        reference = phase t
        moving    = phase t+1

    The network estimates a backward/sampling field defined on the reference
    grid that warps moving -> reference.
    """

    def __init__(
        self,
        data_dir: str | Path,
        phantom: str,
        fr: int = 36,
        phases: Sequence[int] = tuple(range(1, 41)),
        pair_within_phases: Optional[Sequence[int]] = None,
        phase_masks: Optional[Dict[int, np.ndarray]] = None,
        cfg: Config = Config(),
    ):
        self.data_dir = Path(data_dir)
        self.phantom = phantom
        self.fr = int(fr)
        self.cfg = cfg
        self.phase_masks = phase_masks

        if pair_within_phases is None:
            pair_with_phases = list(phases)
        else:
            pair_with_phases = list(pair_within_phases)

        self.image_paths: Dict[int, Path] = {}
        for ph in phases:
            matches = sorted(self.data_dir.glob(f"{phantom}_ph{ph}_fr{fr}.i"))
            if not matches:
                raise FileNotFoundError(
                    f"Missing {phantom}_ph{ph}_fr{fr}.i in {self.data_dir}"
                )
            self.image_paths[ph] = matches[0]

        # Only form adjacent pairs fully contained within the requested set.
        allowed = set(pair_with_phases)
        self.pairs: List[Tuple[int, int]] = []
        for ph in sorted(pair_with_phases):
            nxt = ph + 1
            if nxt in allowed and nxt in self.image_paths:
                self.pairs.append((ph, nxt))

        if not self.pairs:
            raise RuntimeError("No valid adjacent pairs were created.")

        self._cache: Dict[int, np.ndarray] = {}

        # Validate header against expected geometry using the first file.
        first = next(iter(self.image_paths.values()))
        hdr = find_hdr_for_i(first)
        if hdr is not None:
            info = parse_interfile_header(hdr)
            if info.get("matrix_xyz") is not None:
                if tuple(info["matrix_xyz"]) != tuple(cfg.matrix_xyz):
                    raise ValueError(
                        f"Header matrix {info['matrix_xyz']} != expected {cfg.matrix_xyz}"
                    )
            if info.get("spacing_xyz_mm") is not None:
                sp = tuple(float(x) for x in info["spacing_xyz_mm"])
                if not np.allclose(sp, cfg.spacing_xyz_mm, atol=1e-4):
                    raise ValueError(
                        f"Header spacing {sp} != expected {cfg.spacing_xyz_mm}"
                    )

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_phase(self, phase: int) -> np.ndarray:
        if phase not in self._cache:
            dhw = read_i_dhw(
                self.image_paths[phase],
                self.cfg.matrix_xyz,
                expected_dtype=np.dtype("<f4"),
            )
            # Raw data are already (Z,Y,X); crop Z on axis 0.
            dhw = dhw[self.cfg.keep_z_start:self.cfg.keep_z_stop, :, :]
            self._cache[phase] = dhw.astype(np.float32, copy=False)
        return self._cache[phase]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        vals = x[np.isfinite(x)]
        vals = vals[vals > 0]
        if vals.size == 0:
            return np.zeros_like(x, dtype=np.float32)

        lo = np.percentile(vals, self.cfg.norm_low_percentile)
        hi = np.percentile(vals, self.cfg.norm_high_percentile)
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)

        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int):
        ref_ph, mov_ph = self.pairs[idx]
        ref = self._normalize(self._load_phase(ref_ph))
        mov = self._normalize(self._load_phase(mov_ph))

        if self.phase_masks is None:
            ref_mask = np.ones_like(ref, dtype=np.float32)
            mov_mask = np.ones_like(mov, dtype=np.float32)
        else:
            ref_mask = mask_for_phase(ref_ph, self.phase_masks)
            mov_mask = mask_for_phase(mov_ph, self.phase_masks)

        return (
            torch.from_numpy(ref[None]),
            torch.from_numpy(mov[None]),
            torch.from_numpy(ref_mask[None]),
            torch.from_numpy(mov_mask[None]),
            {
                "ref_phase": ref_ph,
                "mov_phase": mov_ph,
                "phantom": self.phantom,
                "fr": self.fr,
                "time_ref_s": (ref_ph - 1) * self.cfg.microframe_seconds,
                "time_mov_s": (mov_ph - 1) * self.cfg.microframe_seconds,
            },
        )


# ============================================================================
# Flow utilities
# ============================================================================

def make_base_grid(
    D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    z = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


def flow_voxels_to_normalized(
    flow_xyz: torch.Tensor,
    shape_dhw: Tuple[int, int, int],
) -> torch.Tensor:
    """
    flow_xyz channels = ux,uy,uz in voxel units.
    Returns normalized grid offsets in x,y,z order.
    """
    D, H, W = shape_dhw
    sx = 2.0 / max(W - 1, 1)
    sy = 2.0 / max(H - 1, 1)
    sz = 2.0 / max(D - 1, 1)

    dx = flow_xyz[:, 0] * sx
    dy = flow_xyz[:, 1] * sy
    dz = flow_xyz[:, 2] * sz
    return torch.stack([dx, dy, dz], dim=-1)


def warp_reference_from_moving(
    moving: torch.Tensor,
    backward_flow: torch.Tensor,
) -> torch.Tensor:
    """
    moving:       (B,C,D,H,W)
    backward_flow (B,3,D,H,W), defined on reference grid.

    Output:
        warped(reference grid) = moving(reference + backward_flow(reference))
    """
    B, C, D, H, W = moving.shape
    base = make_base_grid(D, H, W, moving.device, moving.dtype)
    base = base.unsqueeze(0).expand(B, -1, -1, -1, -1)
    offset = flow_voxels_to_normalized(
        backward_flow.to(moving.dtype), (D, H, W)
    )
    grid = base + offset
    return F.grid_sample(
        moving,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def flow_magnitude_mm(
    flow: torch.Tensor,
    spacing_xyz_mm: Tuple[float, float, float],
) -> torch.Tensor:
    sx, sy, sz = spacing_xyz_mm
    ux = flow[:, 0]
    uy = flow[:, 1]
    uz = flow[:, 2]
    return torch.sqrt(
        (ux * sx) ** 2 + (uy * sy) ** 2 + (uz * sz) ** 2 + 1e-12
    )


def spatial_gradient_3d(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx, dy, dz


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx, dy, dz = spatial_gradient_3d(flow)
    return (
        dx.abs().mean()
        + dy.abs().mean()
        + dz.abs().mean()
    ) / 3.0


def masked_l1(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = (a - b).abs()
    if mask is None:
        return diff.mean()
    m = mask.expand_as(diff).float()
    denom = m.sum().clamp_min(1.0)
    return (diff * m).sum() / denom


def image_gradient_loss(
    reference: torch.Tensor,
    warped: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rdx, rdy, rdz = spatial_gradient_3d(reference)
    wdx, wdy, wdz = spatial_gradient_3d(warped)

    if mask is None:
        return (
            (rdx - wdx).abs().mean()
            + (rdy - wdy).abs().mean()
            + (rdz - wdz).abs().mean()
        ) / 3.0

    mdx = mask[:, :, :, :, 1:].float()
    mdy = mask[:, :, :, 1:, :].float()
    mdz = mask[:, :, 1:, :, :].float()

    return (
        masked_l1(rdx, wdx, mdx)
        + masked_l1(rdy, wdy, mdy)
        + masked_l1(rdz, wdz, mdz)
    ) / 3.0


# ============================================================================
# Attention-based network
# ============================================================================

class ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock3D(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.conv = ConvBlock3D(cout + skip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=True
            )
        return self.conv(torch.cat([x, skip], dim=1))


class SpatialCrossAttention3D(nn.Module):
    """
    Cross-attention at the bottleneck.

    Token count for the current 160x144x144 input after three/four
    downsampling operations remains manageable.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.channels = channels
        self.norm_ref = nn.LayerNorm(channels)
        self.norm_mov = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.proj = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, ref: torch.Tensor, mov: torch.Tensor):
        B, C, D, H, W = ref.shape
        N = D * H * W

        r = ref.flatten(2).transpose(1, 2)  # B,N,C
        m = mov.flatten(2).transpose(1, 2)

        # moving queries reference; this gives moving/reference correspondence.
        q = self.norm_mov(m)
        k = self.norm_ref(r)
        v = r

        attended, _ = self.attn(q, k, v, need_weights=False)

        attended = attended.transpose(1, 2).reshape(B, C, D, H, W)
        mov_out = mov + attended

        return self.proj(torch.cat([ref, mov_out], dim=1))


class MotionCorrectionNet(nn.Module):
    """
    Predicts a backward/sampling flow on the reference grid.

    forward(reference, moving):
        flow[:,0] = x sampling displacement in voxels
        flow[:,1] = y sampling displacement in voxels
        flow[:,2] = z sampling displacement in voxels
    """

    def __init__(self, base_channels: int = 16, attention_heads: int = 4):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock3D(2, c)
        self.enc2 = ConvBlock3D(c, 2*c, stride=2)
        self.enc3 = ConvBlock3D(2*c, 4*c, stride=2)
        self.enc4 = ConvBlock3D(4*c, 8*c, stride=2)
        self.enc5 = ConvBlock3D(8*c, 8*c, stride=2)

        self.attn = SpatialCrossAttention3D(8*c, heads=attention_heads)

        self.up4 = UpBlock3D(8*c, 8*c, 4*c)
        self.up3 = UpBlock3D(4*c, 4*c, 2*c)
        self.up2 = UpBlock3D(2*c, 2*c, c)
        self.up1 = UpBlock3D(c, c, c)

        self.flow_head = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c, 3, 3, padding=1),
        )

        # Start near zero deformation.
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def forward(self, reference: torch.Tensor, moving: torch.Tensor):
        x = torch.cat([reference, moving], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.enc5(e4)

        b = self.attn(e4, b)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        flow = self.flow_head(d1)

        # Exact output-size safeguard.
        if flow.shape[2:] != reference.shape[2:]:
            flow = F.interpolate(
                flow,
                size=reference.shape[2:],
                mode="trilinear",
                align_corners=True,
            )

        return flow


# ============================================================================
# Training / validation
# ============================================================================

def compute_self_supervised_loss(
    reference: torch.Tensor,
    moving: torch.Tensor,
    reference_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    flow: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    warped = warp_reference_from_moving(moving, flow)

    # Reference myocardium mask is the natural evaluation/training ROI.
    # The moving mask is warped to the reference grid to identify valid overlap.
    warped_moving_mask = warp_reference_from_moving(
        moving_mask, flow
    ).clamp(0, 1)

    overlap = reference_mask * warped_moving_mask

    # Always retain a whole-image photometric term; add myocardium weighting.
    photo_whole = masked_l1(reference, warped, None)
    photo_heart = masked_l1(reference, warped, reference_mask)
    photo_overlap = masked_l1(reference, warped, overlap)

    photo = photo_whole + cfg.w_mask_photo * 0.5 * (
        photo_heart + photo_overlap
    )

    smooth = smoothness_loss(flow)
    grad = image_gradient_loss(reference, warped, reference_mask)

    total = (
        cfg.w_photo * photo
        + cfg.w_smooth * smooth
        + cfg.w_grad * grad
    )

    stats = {
        "loss": float(total.detach().cpu()),
        "photo_whole": float(photo_whole.detach().cpu()),
        "photo_heart": float(photo_heart.detach().cpu()),
        "photo_overlap": float(photo_overlap.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "gradient": float(grad.detach().cpu()),
    }
    return total, stats, warped


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    cfg: Config,
    train: bool,
) -> Dict[str, float]:
    model.train(train)

    sums: Dict[str, float] = {}
    n = 0

    for reference, moving, ref_mask, mov_mask, meta in loader:
        reference = reference.to(device, non_blocking=True)
        moving = moving.to(device, non_blocking=True)
        ref_mask = ref_mask.to(device, non_blocking=True)
        mov_mask = mov_mask.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = bool(cfg.use_amp and device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=amp_enabled,
        ):
            flow = model(reference, moving)
            loss, stats, _ = compute_self_supervised_loss(
                reference, moving, ref_mask, mov_mask, flow, cfg
            )

        if train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        bs = reference.shape[0]
        n += bs
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v * bs

    return {k: v / max(n, 1) for k, v in sums.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val: float,
    cfg: Config,
) -> None:
    """Save a complete resumable training state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val": float(best_val),
        "config": asdict(cfg),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    # Atomic replacement is useful if a Colab session is interrupted during
    # the checkpoint write. The previous complete latest checkpoint remains
    # intact until the temporary file is successfully written.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> Tuple[int, float, List[dict]]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    _restore_rng_state(ckpt)
    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val", float("inf")))
    return epoch, best_val, []


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    cfg: Config,
    resume_path: Optional[Path] = None,
    explicit_checkpoint: Optional[Path] = None,
) -> Path:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.use_amp and device.type == "cuda")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"
    latest_path = out_dir / "latest_checkpoint.pt"
    history_path = out_dir / "training_history.json"

    history: List[dict] = []
    start_epoch = 1

    resume_target = explicit_checkpoint if explicit_checkpoint is not None else resume_path
    if resume_target is not None and resume_target.exists():
        ckpt = torch.load(resume_target, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        _restore_rng_state(ckpt)
        last_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        start_epoch = last_epoch + 1
        print(f"Resumed checkpoint epoch {last_epoch}; continuing at epoch {start_epoch}.")

        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                history = [r for r in history if int(r.get("epoch", 0)) <= last_epoch]
            except Exception:
                history = []

    if start_epoch > cfg.epochs:
        print(f"Checkpoint already reached epoch {start_epoch-1}, target is {cfg.epochs}.")
        if best_path.exists():
            return best_path
        if latest_path.exists():
            return latest_path
        return final_path

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        train_stats = run_epoch(
            model, train_loader, optimizer, scaler, device, cfg, train=True
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model, val_loader, None, None, device, cfg, train=False
            )

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "seconds": time.time() - t0,
        }
        history.append(row)
        val_loss = val_stats["loss"]

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_val, cfg)

        # Rolling checkpoint: only ONE file is kept, so this does not create
        # 100 large model files. It is the key checkpoint for Colab resume.
        save_checkpoint(latest_path, model, optimizer, scaler, epoch, best_val, cfg)

        if cfg.save_every_epoch:
            save_checkpoint(
                out_dir / f"epoch_{epoch:04d}.pt",
                model, optimizer, scaler, epoch, best_val, cfg
            )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train={train_stats['loss']:.6f} | "
            f"val={val_stats['loss']:.6f} | "
            f"{row['seconds']:.1f}s"
        )

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # Final checkpoint is saved separately from latest.
    save_checkpoint(
        final_path, model, optimizer, scaler, cfg.epochs, best_val, cfg
    )

    print(f"Best checkpoint:   {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Final checkpoint:  {final_path}")
    return best_path


# ============================================================================
# Arbitrary-reference inference
# ============================================================================

@torch.no_grad()
def predict_flow(
    model: nn.Module,
    reference_dhw: np.ndarray,
    moving_dhw: np.ndarray,
    reference_mask_dhw: Optional[np.ndarray],
    device: torch.device,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align ANY moving frame to ANY selected reference frame.

    Returns:
        flow_backward_vox: (3,D,H,W)
        warped_moving:     (D,H,W)
    """
    model.eval()

    def norm(x):
        vals = x[np.isfinite(x)]
        vals = vals[vals > 0]
        if vals.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        lo = np.percentile(vals, cfg.norm_low_percentile)
        hi = np.percentile(vals, cfg.norm_high_percentile)
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)

    ref = torch.from_numpy(norm(reference_dhw)[None, None]).float().to(device)
    mov = torch.from_numpy(norm(moving_dhw)[None, None]).float().to(device)

    flow = model(ref, mov)
    warped = warp_reference_from_moving(mov, flow)

    return (
        flow[0].detach().cpu().numpy().astype(np.float32),
        warped[0, 0].detach().cpu().numpy().astype(np.float32),
    )


def load_image_dhw(
    data_dir: Path,
    phantom: str,
    phase: int,
    fr: int,
    cfg: Config,
) -> np.ndarray:
    p = data_dir / f"{phantom}_ph{phase}_fr{fr}.i"
    if not p.exists():
        raise FileNotFoundError(p)

    dhw = read_i_dhw(p, cfg.matrix_xyz, expected_dtype=np.dtype("<f4"))
    return dhw[cfg.keep_z_start:cfg.keep_z_stop, :, :].astype(np.float32, copy=False)


# ============================================================================
# Flow metric utilities
# ============================================================================

def masked_flow_metrics(
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    mask_dhw: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
) -> Dict[str, float]:
    """
    Both fields must have the SAME convention:
        backward/reference-grid displacement, channels ux,uy,uz.

    Reports voxel and mm EPE, median, P95, and component MAE.
    """
    mask = mask_dhw.astype(bool)
    if not np.any(mask):
        raise ValueError("Metric mask is empty.")

    diff = pred_flow.astype(np.float64) - gt_flow.astype(np.float64)
    epe_vox = np.sqrt(np.sum(diff ** 2, axis=0))

    sx, sy, sz = spacing_xyz_mm
    diff_mm = diff.copy()
    diff_mm[0] *= sx
    diff_mm[1] *= sy
    diff_mm[2] *= sz
    epe_mm = np.sqrt(np.sum(diff_mm ** 2, axis=0))

    vals_vox = epe_vox[mask]
    vals_mm = epe_mm[mask]

    return {
        "mean_epe_vox": float(np.mean(vals_vox)),
        "median_epe_vox": float(np.median(vals_vox)),
        "p95_epe_vox": float(np.percentile(vals_vox, 95)),
        "mean_epe_mm": float(np.mean(vals_mm)),
        "median_epe_mm": float(np.median(vals_mm)),
        "p95_epe_mm": float(np.percentile(vals_mm, 95)),
        "mae_ux_vox": float(np.mean(np.abs(diff[0][mask]))),
        "mae_uy_vox": float(np.mean(np.abs(diff[1][mask]))),
        "mae_uz_vox": float(np.mean(np.abs(diff[2][mask]))),
        "mae_ux_mm": float(np.mean(np.abs(diff_mm[0][mask]))),
        "mae_uy_mm": float(np.mean(np.abs(diff_mm[1][mask]))),
        "mae_uz_mm": float(np.mean(np.abs(diff_mm[2][mask]))),
    }


def displacement_summary(
    flow: np.ndarray,
    mask_dhw: np.ndarray,
    spacing_xyz_mm: Tuple[float, float, float],
) -> Dict[str, float]:
    ux, uy, uz = flow
    sx, sy, sz = spacing_xyz_mm

    mag_vox = np.sqrt(ux**2 + uy**2 + uz**2)
    mag_mm = np.sqrt(
        (ux*sx)**2 + (uy*sy)**2 + (uz*sz)**2
    )

    m = mask_dhw.astype(bool)
    if not np.any(m):
        raise ValueError("Displacement mask is empty.")

    return {
        "mean_displacement_vox": float(np.mean(mag_vox[m])),
        "median_displacement_vox": float(np.median(mag_vox[m])),
        "p95_displacement_vox": float(np.percentile(mag_vox[m], 95)),
        "max_displacement_vox": float(np.max(mag_vox[m])),
        "mean_displacement_mm": float(np.mean(mag_mm[m])),
        "median_displacement_mm": float(np.median(mag_mm[m])),
        "p95_displacement_mm": float(np.percentile(mag_mm[m], 95)),
        "max_displacement_mm": float(np.max(mag_mm[m])),
    }


# ============================================================================
# XCAT ground-truth loading
# ============================================================================

def load_xcat_field_raw(
    path: Path,
    shape_xyz: Tuple[int, int, int],
    dtype: np.dtype = np.dtype("<f4"),
    channels_first: bool = True,
) -> np.ndarray:
    """
    Load a dense XCAT vector field from .npy/.npz/raw binary.

    Accepted resulting shapes:
        (3,X,Y,Z)
        (X,Y,Z,3)

    For .npy/.npz, shape is checked explicitly.
    For raw binary, exactly 3*X*Y*Z values are required.

    This function deliberately does NOT guess a vector-file format.
    """
    X, Y, Z = shape_xyz

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path)
        if "flow" in z:
            arr = z["flow"]
        elif "dvf" in z:
            arr = z["dvf"]
        elif len(z.files) == 1:
            arr = z[z.files[0]]
        else:
            raise ValueError(
                f"{path}: NPZ must contain 'flow', 'dvf', or one array."
            )
    else:
        arr = np.fromfile(path, dtype=dtype)
        if arr.size != 3 * X * Y * Z:
            raise ValueError(
                f"{path}: raw field has {arr.size} values; expected "
                f"{3*X*Y*Z}. The exact XCAT vector output format must be "
                f"converted to a dense 3-channel field first."
            )
        arr = arr.reshape((3, X, Y, Z))

    arr = np.asarray(arr, dtype=np.float32)

    if arr.shape == (3, X, Y, Z):
        return arr
    if arr.shape == (X, Y, Z, 3):
        return np.moveaxis(arr, -1, 0)

    raise ValueError(
        f"{path}: field shape {arr.shape} is not one of "
        f"(3,X,Y,Z)={(3,X,Y,Z)} or (X,Y,Z,3)={(X,Y,Z,3)}."
    )


def crop_xyz_field_to_dhw(
    field_cxyz: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Input: (3,X,Y,Z)
    Output: (3,D,H,W) = (3,Z,Y,X)
    """
    cropped = field_cxyz[
        :,
        :,
        :,
        cfg.keep_z_start:cfg.keep_z_stop,
    ]
    return np.transpose(cropped, (0, 3, 2, 1)).astype(np.float32)


def invert_forward_field_fixed_point(
    forward_xyz: np.ndarray,
    iterations: int = 30,
) -> np.ndarray:
    """
    Numerically invert a forward deformation:
        y = x + u_forward(x)

    The returned inverse field v is defined on y:
        x = y + v(y)

    Fixed-point iteration:
        v_{n+1}(y) = -u_forward(y + v_n(y))

    This is only appropriate for a sufficiently smooth, invertible field.
    It is included so that an XCAT forward DVF can be compared to the
    network's backward/reference-grid sampling field without image
    registration.

    Input/output: (3,D,H,W), voxel units.
    """
    if forward_xyz.ndim != 4 or forward_xyz.shape[0] != 3:
        raise ValueError("forward_xyz must be (3,D,H,W)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = torch.from_numpy(forward_xyz[None]).float().to(device)
    _, _, D, H, W = f.shape

    v = torch.zeros_like(f)

    base = make_base_grid(D, H, W, device, f.dtype).unsqueeze(0)

    for _ in range(iterations):
        offset = flow_voxels_to_normalized(v, (D, H, W))
        grid = base + offset

        sampled = F.grid_sample(
            f,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        v_new = -sampled

        if torch.max(torch.abs(v_new - v)).item() < 1e-4:
            v = v_new
            break
        v = v_new

    return v[0].cpu().numpy().astype(np.float32)


def get_xcat_gt_pair(
    gt_dir: Path,
    phantom: str,
    reference_phase: int,
    moving_phase: int,
    cfg: Config,
) -> np.ndarray:
    """
    Preferred GT representation:
        one absolute XCAT forward field per temporal phase, relative to the
        same XCAT reference coordinate system.

    If the fields are absolute/reference-to-phase fields:
        F_ref->ref = 0
        F_ref->moving = field[moving_phase]

    The pairwise forward field can be obtained by composing/inverting those
    absolute fields, but this requires the actual XCAT coordinate convention.

    Therefore this function intentionally supports the safest case:
        a pairwise file named:
            {phantom}_ph{reference}_to_ph{moving}.npy

    The pairwise file must be a dense field in (3,X,Y,Z) or (X,Y,Z,3).
    """
    pair_name = f"{phantom}_ph{reference_phase:02d}_to_ph{moving_phase:02d}.npy"
    path = gt_dir / pair_name
    if not path.exists():
        raise FileNotFoundError(
            f"True XCAT GT pair not found: {path}\n"
            "Export/convert the XCAT vector-mode output to this explicit "
            "pairwise dense format before running quantitative GT evaluation."
        )

    forward_xyz = load_xcat_field_raw(
        path,
        cfg.xcat_gt_shape_xyz,
        dtype=np.dtype(cfg.xcat_gt_dtype),
        channels_first=cfg.xcat_gt_channels_first,
    )

    forward_dhw = crop_xyz_field_to_dhw(forward_xyz, cfg)

    if cfg.xcat_gt_direction.lower() == "forward":
        # XCAT field is reference -> moving; invert it to the network's
        # backward/reference-grid sampling convention.
        return invert_forward_field_fixed_point(forward_dhw)
    elif cfg.xcat_gt_direction.lower() == "backward":
        return forward_dhw
    else:
        raise ValueError(
            "xcat_gt_direction must be 'forward' or 'backward'."
        )


# ============================================================================
# Gate construction
# ============================================================================

def gate_phase_groups() -> Dict[int, List[int]]:
    return {
        k: [k, k+8, k+16, k+24, k+32]
        for k in range(1, 9)
    }


def save_gate_groups(out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(gate_phase_groups(), f, indent=2)


@torch.no_grad()
def make_motion_corrected_gates(
    model: nn.Module,
    data_dir: Path,
    phantom: str,
    fr: int,
    phase_masks: Dict[int, np.ndarray],
    device: torch.device,
    cfg: Config,
    out_dir: Path,
) -> None:
    """
    Demonstration only.

    For each cardiac gate k:
        reference = first occurrence k
        align k+8, k+16, k+24, k+32 -> reference
        average the five aligned images.

    The conventional gate is simply the average of the five uncorrected
    microframes. It is NOT treated as a motion-free ground truth.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    results = {}

    for gate, phases in gate_phase_groups().items():
        ref_phase = phases[cfg.gate_reference_occurrence]
        ref = load_image_dhw(data_dir, phantom, ref_phase, fr, cfg)

        conventional = []
        corrected = []

        for ph in phases:
            img = load_image_dhw(data_dir, phantom, ph, fr, cfg)
            conventional.append(img)

            if ph == ref_phase:
                corrected.append(img)
                continue

            ref_mask = mask_for_phase(ref_phase, phase_masks)

            flow, warped = predict_flow(
                model=model,
                reference_dhw=ref,
                moving_dhw=img,
                reference_mask_dhw=ref_mask,
                device=device,
                cfg=cfg,
            )
            corrected.append(warped)

            np.save(
                out_dir / f"{phantom}_gate{gate:02d}_ph{ph:02d}_to_ref_ph{ref_phase:02d}_flow_backward_vox.npy",
                flow,
            )

        g_uncorrected = np.mean(np.stack(conventional), axis=0)
        g_corrected = np.mean(np.stack(corrected), axis=0)

        np.save(out_dir / f"{phantom}_gate{gate:02d}_uncorrected.npy", g_uncorrected)
        np.save(out_dir / f"{phantom}_gate{gate:02d}_motion_corrected.npy", g_corrected)

        results[gate] = {
            "phases": phases,
            "reference_phase": ref_phase,
        }

    with open(out_dir / "gate_metadata.json", "w") as f:
        json.dump(results, f, indent=2)


# ============================================================================
# Valid-volume masks / arbitrary pair inference
# ============================================================================

def valid_pet_mask(reference_dhw: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Exclude the large zero/background region from whole-volume flow summaries."""
    x = np.asarray(reference_dhw)
    return np.isfinite(x) & (x > float(threshold))


@torch.no_grad()
def run_arbitrary_pair_inference(
    model: nn.Module,
    data_dir: Path,
    phantom: str,
    reference_phase: int,
    moving_phase: int,
    fr: int,
    phase_masks: Dict[int, np.ndarray],
    device: torch.device,
    cfg: Config,
    out_dir: Path,
) -> Dict[str, object]:
    """
    Align any selected phase to any other selected phase.

    Example: reference=16, moving=26 is scientifically meaningful as a
    non-adjacent pairwise registration/motion-estimation experiment. It is
    not an XCAT-GT experiment and should not be described as such.
    """
    if reference_phase == moving_phase:
        raise ValueError("Reference and moving phases must be different.")

    out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_image_dhw(data_dir, phantom, reference_phase, fr, cfg)
    mov = load_image_dhw(data_dir, phantom, moving_phase, fr, cfg)
    ref_mask = mask_for_phase(reference_phase, phase_masks)

    flow, warped = predict_flow(model, ref, mov, ref_mask, device, cfg)

    stem = f"{phantom}_ph{moving_phase:02d}_to_ph{reference_phase:02d}"
    np.save(out_dir / f"{stem}_pred_backward_vox.npy", flow)
    np.save(out_dir / f"{stem}_warped.npy", warped)
    np.save(out_dir / f"{stem}_magnitude_mm.npy", np.sqrt(
        (flow[0] * cfg.spacing_xyz_mm[0]) ** 2
        + (flow[1] * cfg.spacing_xyz_mm[1]) ** 2
        + (flow[2] * cfg.spacing_xyz_mm[2]) ** 2
    ).astype(np.float32))

    summary = displacement_summary(flow, ref_mask, cfg.spacing_xyz_mm)
    metadata = {
        "phantom": phantom,
        "reference_phase": reference_phase,
        "moving_phase": moving_phase,
        "fr": fr,
        "flow_convention": "backward/sampling field on reference grid; channels ux,uy,uz",
        "summary_myocardium": summary,
    }
    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# ============================================================================
# Test all 40 p150 states
# ============================================================================

@torch.no_grad()
def evaluate_test_pairs(
    model: nn.Module,
    data_dir: Path,
    phantom: str,
    fr: int,
    phase_masks: Dict[int, np.ndarray],
    device: torch.device,
    cfg: Config,
    out_dir: Path,
) -> List[Dict[str, object]]:
    """
    Evaluate all adjacent p150 pairs:
        1->2, 2->3, ..., 39->40.

    If true XCAT GT pairwise fields are present, quantitative EPE is reported.
    Otherwise the script reports displacement summaries only and explicitly
    marks GT metrics as unavailable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    rows = []

    for ref_phase in range(1, 40):
        moving_phase = ref_phase + 1

        ref = load_image_dhw(data_dir, phantom, ref_phase, fr, cfg)
        mov = load_image_dhw(data_dir, phantom, moving_phase, fr, cfg)

        ref_mask = mask_for_phase(ref_phase, phase_masks)
        mov_mask = mask_for_phase(moving_phase, phase_masks)

        flow, warped = predict_flow(
            model, ref, mov, ref_mask, device, cfg
        )

        whole_mask = valid_pet_mask(ref, cfg.valid_image_threshold).astype(np.float32)

        whole_summary = displacement_summary(
            flow, whole_mask, cfg.spacing_xyz_mm
        )
        heart_summary = displacement_summary(
            flow, ref_mask, cfg.spacing_xyz_mm
        )

        row = {
            "phantom": phantom,
            "fr": fr,
            "reference_phase": ref_phase,
            "moving_phase": moving_phase,
            "whole_image": whole_summary,
            "myocardium": heart_summary,
            "ground_truth_available": False,
        }

        # Optional true XCAT GT.
        if cfg.xcat_gt_dir is not None:
            try:
                gt = get_xcat_gt_pair(
                    Path(cfg.xcat_gt_dir),
                    phantom,
                    ref_phase,
                    moving_phase,
                    cfg,
                )

                row["ground_truth_available"] = True
                row["gt_units"] = "voxel"
                row["whole_image_gt"] = masked_flow_metrics(
                    flow, gt, whole_mask, cfg.spacing_xyz_mm
                )
                row["myocardium_gt"] = masked_flow_metrics(
                    flow, gt, ref_mask, cfg.spacing_xyz_mm
                )

                np.save(
                    out_dir
                    / f"{phantom}_ph{moving_phase:02d}_to_ph{ref_phase:02d}_gt_backward_vox.npy",
                    gt,
                )
            except FileNotFoundError:
                # Missing GT is not silently converted to registration.
                row["ground_truth_error"] = "GT pair not found"
            except Exception as e:
                row["ground_truth_error"] = str(e)

        np.save(
            out_dir
            / f"{phantom}_ph{moving_phase:02d}_to_ph{ref_phase:02d}_pred_backward_vox.npy",
            flow,
        )
        np.save(
            out_dir
            / f"{phantom}_ph{moving_phase:02d}_warped_to_ph{ref_phase:02d}.npy",
            warped,
        )

        rows.append(row)
        print(
            f"Test {phantom}: {ref_phase:02d}->{moving_phase:02d} | "
            f"mean displacement whole={whole_summary['mean_displacement_mm']:.3f} mm | "
            f"myocardium={heart_summary['mean_displacement_mm']:.3f} mm"
        )

    with open(out_dir / f"{phantom}_flow_evaluation.json", "w") as f:
        json.dump(rows, f, indent=2)

    return rows


# ============================================================================
# Optional conventional registration for qualitative MVF presentation
# ============================================================================

def _require_simpleitk():
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required only for optional conventional registration. "
            "In Google Colab run: !pip -q install SimpleITK"
        ) from exc


def _sitk_image_from_dhw(dhw: np.ndarray, spacing_xyz_mm: Tuple[float, float, float]):
    sitk = _require_simpleitk()
    img = sitk.GetImageFromArray(np.asarray(dhw, dtype=np.float32))
    img.SetSpacing(tuple(float(v) for v in spacing_xyz_mm))
    return img


def _sitk_field_to_dhw_vox(field_img, cfg: Config) -> np.ndarray:
    """SimpleITK displacement field -> (3,D,H,W) voxel displacement."""
    sitk = _require_simpleitk()
    arr_zyx3 = sitk.GetArrayFromImage(field_img).astype(np.float32)
    if arr_zyx3.ndim != 4 or arr_zyx3.shape[-1] != 3:
        raise ValueError(f"Unexpected displacement field array shape: {arr_zyx3.shape}")
    # SimpleITK vector components are physical X,Y,Z, i.e. mm.
    field_mm = np.moveaxis(arr_zyx3, -1, 0)  # (3,D,H,W)
    field_vox = field_mm.copy()
    field_vox[0] /= cfg.spacing_xyz_mm[0]
    field_vox[1] /= cfg.spacing_xyz_mm[1]
    field_vox[2] /= cfg.spacing_xyz_mm[2]
    return field_vox.astype(np.float32)


def conventional_demons_registration(
    reference_dhw: np.ndarray,
    moving_dhw: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Qualitative conventional registration only.

    A multi-resolution Demons registration is performed on smoothed PET images.
    The resulting displacement field is defined on the reference/fixed grid and
    is therefore directly comparable in *convention* to the network's backward
    sampling field. It is NOT treated as ground truth.

    Returns:
        field_backward_vox: (3,D,H,W)
        warped_moving: (D,H,W)
    """
    sitk = _require_simpleitk()

    fixed = _sitk_image_from_dhw(reference_dhw, cfg.spacing_xyz_mm)
    moving = _sitk_image_from_dhw(moving_dhw, cfg.spacing_xyz_mm)

    # Robust intensity preparation for registration; this does not alter the
    # PET arrays saved elsewhere. Histogram matching is used only for the
    # registration similarity calculation so that tracer-intensity scaling
    # between frames is less likely to drive the deformation field.
    fixed = sitk.Normalize(fixed)
    moving = sitk.Normalize(moving)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(128)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    if cfg.registration_smoothing_sigma_mm > 0:
        fixed = sitk.SmoothingRecursiveGaussian(fixed, cfg.registration_smoothing_sigma_mm)
        moving = sitk.SmoothingRecursiveGaussian(moving, cfg.registration_smoothing_sigma_mm)

    shrink = max(1, int(cfg.registration_shrink))
    if shrink > 1:
        fixed_reg = sitk.Shrink(fixed, [shrink] * 3)
        moving_reg = sitk.Shrink(moving, [shrink] * 3)
    else:
        fixed_reg, moving_reg = fixed, moving

    # Prefer diffeomorphic Demons for a smooth physically plausible qualitative
    # field; fall back to standard Demons if an older SimpleITK build does not
    # expose the diffeomorphic filter. SimpleITK documents both filters as
    # producing a displacement field mapping the moving image onto the fixed
    # image.
    if hasattr(sitk, "DiffeomorphicDemonsRegistrationFilter"):
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(int(cfg.registration_iterations))
        demons.SetStandardDeviations(float(cfg.registration_update_sigma_mm))
        if hasattr(demons, "SetUpdateFieldStandardDeviations"):
            demons.SetUpdateFieldStandardDeviations(float(cfg.registration_update_sigma_mm))
        if hasattr(demons, "SetMaximumUpdateStepLength"):
            demons.SetMaximumUpdateStepLength(0.5)
        demons.SetMaximumRMSError(1e-4)
        demons.SetIntensityDifferenceThreshold(0.001)
        field_coarse = demons.Execute(fixed_reg, moving_reg)
        registration_method = "DiffeomorphicDemonsRegistrationFilter"
    else:
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(int(cfg.registration_iterations))
        demons.SetStandardDeviations(float(cfg.registration_update_sigma_mm))
        demons.SetMaximumRMSError(1e-4)
        demons.SetIntensityDifferenceThreshold(0.001)
        field_coarse = demons.Execute(fixed_reg, moving_reg)
        registration_method = "DemonsRegistrationFilter"

    # Resample the vector field onto the original fixed grid. Displacements are
    # stored by SimpleITK in physical units (mm), so no manual scale factor is
    # applied during this interpolation.
    reference_full = _sitk_image_from_dhw(reference_dhw, cfg.spacing_xyz_mm)
    field_full = sitk.Resample(
        field_coarse,
        reference_full,
        sitk.Transform(),
        sitk.sitkLinear,
        (0.0, 0.0, 0.0),
        sitk.sitkVectorFloat32,
    )

    # Field is defined on fixed/reference grid and maps fixed points into the
    # moving image for sampling: exactly the convention used by grid_sample.
    flow_vox = _sitk_field_to_dhw_vox(field_full, cfg)

    transform = sitk.DisplacementFieldTransform(field_full)
    warped_img = sitk.Resample(
        _sitk_image_from_dhw(moving_dhw, cfg.spacing_xyz_mm),
        reference_full,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    warped_dhw = sitk.GetArrayFromImage(warped_img).astype(np.float32)

    return flow_vox, warped_dhw


def save_flow_presentation(
    reference_dhw: np.ndarray,
    flow_vox: np.ndarray,
    out_png: Path,
    cfg: Config,
    title: str,
    mask_dhw: Optional[np.ndarray] = None,
) -> None:
    """Save a compact three-plane magnitude/quiver MVF figure."""
    import matplotlib.pyplot as plt

    mag_mm = np.sqrt(
        (flow_vox[0] * cfg.spacing_xyz_mm[0]) ** 2
        + (flow_vox[1] * cfg.spacing_xyz_mm[1]) ** 2
        + (flow_vox[2] * cfg.spacing_xyz_mm[2]) ** 2
    )

    if mask_dhw is not None and np.any(mask_dhw):
        valid = mask_dhw.astype(bool)
        zyx = np.argwhere(valid)
        zc, yc, xc = np.median(zyx, axis=0).astype(int)
    else:
        zc, yc, xc = [n // 2 for n in reference_dhw.shape]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    step = 8

    # Axial: z fixed, x/y arrows.
    ax = axes[0]
    ax.imshow(reference_dhw[zc], cmap="gray")
    yy, xx = np.mgrid[0:reference_dhw.shape[1]:step, 0:reference_dhw.shape[2]:step]
    ax.quiver(xx, yy, flow_vox[0, zc, ::step, ::step], flow_vox[1, zc, ::step, ::step],
              mag_mm[zc, ::step, ::step], cmap="turbo", angles="xy", scale_units="xy", scale=1)
    ax.set_title(f"Axial z={zc}")
    ax.axis("off")

    # Coronal: y fixed, x/z arrows.
    ax = axes[1]
    ax.imshow(reference_dhw[:, yc, :], cmap="gray", aspect="auto")
    zz, xx = np.mgrid[0:reference_dhw.shape[0]:step, 0:reference_dhw.shape[2]:step]
    ax.quiver(xx, zz, flow_vox[0, ::step, yc, ::step], flow_vox[2, ::step, yc, ::step],
              mag_mm[::step, yc, ::step], cmap="turbo", angles="xy", scale_units="xy", scale=1)
    ax.set_title(f"Coronal y={yc}")
    ax.axis("off")

    # Sagittal: x fixed, y/z arrows.
    ax = axes[2]
    ax.imshow(reference_dhw[:, :, xc], cmap="gray", aspect="auto")
    zz, yy = np.mgrid[0:reference_dhw.shape[0]:step, 0:reference_dhw.shape[1]:step]
    ax.quiver(yy, zz, flow_vox[1, ::step, ::step, xc], flow_vox[2, ::step, ::step, xc],
              mag_mm[::step, ::step, xc], cmap="turbo", angles="xy", scale_units="xy", scale=1)
    ax.set_title(f"Sagittal x={xc}")
    ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_conventional_registration_pair(
    data_dir: Path,
    phantom: str,
    reference_phase: int,
    moving_phase: int,
    fr: int,
    phase_masks: Dict[int, np.ndarray],
    cfg: Config,
    out_dir: Path,
) -> Dict[str, object]:
    ref = load_image_dhw(data_dir, phantom, reference_phase, fr, cfg)
    mov = load_image_dhw(data_dir, phantom, moving_phase, fr, cfg)
    flow_vox, warped = conventional_demons_registration(ref, mov, cfg)
    mask = mask_for_phase(reference_phase, phase_masks)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{phantom}_ph{moving_phase:02d}_to_ph{reference_phase:02d}_demons"
    np.save(out_dir / f"{stem}_backward_vox.npy", flow_vox)
    np.save(out_dir / f"{stem}_warped.npy", warped)
    np.save(out_dir / f"{stem}_magnitude_mm.npy", np.sqrt(
        (flow_vox[0] * cfg.spacing_xyz_mm[0]) ** 2
        + (flow_vox[1] * cfg.spacing_xyz_mm[1]) ** 2
        + (flow_vox[2] * cfg.spacing_xyz_mm[2]) ** 2
    ).astype(np.float32))

    save_flow_presentation(
        ref, flow_vox, out_dir / f"{stem}_MVF_3plane.png", cfg,
        title=f"Conventional Demons MVF: {phantom} ph{moving_phase} -> ph{reference_phase}",
        mask_dhw=mask,
    )

    summary = displacement_summary(flow_vox, mask, cfg.spacing_xyz_mm)
    meta = {
        "method": registration_method,
        "phantom": phantom,
        "reference_phase": reference_phase,
        "moving_phase": moving_phase,
        "fr": fr,
        "flow_convention": "backward/reference-grid sampling field",
        "field_units_saved": "voxels",
        "field_physical_units": "mm internally",
        "NOT_ground_truth": True,
        "myocardium_displacement_summary": summary,
    }
    with open(out_dir / f"{stem}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def parse_pair_specs(spec: str) -> List[Tuple[int, int]]:
    """Parse '16:26,1:5' as [(16,26),(1,5)] (reference:moving)."""
    pairs = []
    if not spec:
        return pairs
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.fullmatch(r"(\d+)\s*[:\-]\s*(\d+)", item)
        if not m:
            raise ValueError(f"Invalid pair specification '{item}'. Use reference:moving, e.g. 16:26.")
        pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


# ============================================================================
# Main
# ============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser()

    # Colab / Google Drive defaults requested by the user.
    p.add_argument("--train_dir", "--p80_dir", dest="train_dir", default="/content/Output_p80_i")
    p.add_argument("--test_dir", "--p150_dir", dest="test_dir", default="/content/Output_p150_i")
    p.add_argument("--drive_mount", default="/content/drive")
    p.add_argument("--drive_root", default="/content/drive/MyDrive")
    p.add_argument("--output_dir", default="/content/drive/MyDrive/self_supervised_results")
    p.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/self_supervised_checkpoints")
    p.add_argument("--inference_root", default="/content/drive/MyDrive/self_supervised_results/p150")

    p.add_argument("--fr", type=int, default=36)
    p.add_argument("--mask_dir", default=None)
    p.add_argument("--generate_masks", action="store_true")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--no_resume", action="store_true", help="Do not auto-resume from latest checkpoint.")

    p.add_argument("--xcat_gt_dir", default=None)
    p.add_argument("--xcat_gt_direction", choices=["forward", "backward"], default="forward")

    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint to load before inference/training.")

    # Arbitrary reference/moving pairs, e.g. --arbitrary_pairs 16:26,16:8
    p.add_argument("--arbitrary_pairs", default="16:26")
    p.add_argument("--skip_arbitrary_inference", action="store_true")

    # Conventional registration is OFF by default. It is only for qualitative
    # comparison/presentation and never enters the training loss or GT metrics.
    p.add_argument("--run_registration", action="store_true")
    p.add_argument("--registration_pairs", default="16:26")
    p.add_argument("--registration_iterations", type=int, default=80)
    p.add_argument("--registration_shrink", type=int, default=2)

    return p


def _maybe_mount_google_drive(drive_mount: str) -> None:
    """Mount Drive when running inside Colab; no-op elsewhere."""
    if not os.path.exists("/content"):
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount(drive_mount, force_remount=False)
        print(f"Google Drive available at {drive_mount}")
    except Exception as e:
        print(f"Google Drive mount skipped/unavailable: {e}")


def _restore_rng_state(ckpt: dict) -> None:
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        except Exception:
            pass


def main():
    args = build_arg_parser().parse_args()
    _maybe_mount_google_drive(args.drive_mount)

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        use_amp=args.use_amp,
        xcat_gt_dir=args.xcat_gt_dir,
        xcat_gt_direction=args.xcat_gt_direction,
        registration_enabled=args.run_registration,
        registration_iterations=args.registration_iterations,
        registration_shrink=args.registration_shrink,
        resume=not args.no_resume,
    )

    set_seed(cfg.seed)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    inference_root = Path(args.inference_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Raw storage convention: {cfg.raw_storage_order.upper()} -> tensor (D,H,W)")
    print(f"Header matrix XYZ: {cfg.matrix_xyz}; cropped tensor shape: "
          f"({cfg.keep_z_stop-cfg.keep_z_start},{cfg.matrix_xyz[1]},{cfg.matrix_xyz[0]})")
    print(f"Spacing XYZ [mm]: {cfg.spacing_xyz_mm}")

    # Save an explicit geometry note so there is no ambiguity later.
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(output_dir / "DATA_ORIENTATION.txt", "w") as f:
        f.write(
            "Interfile header matrix order: X,Y,Z = 144,144,355\n"
            "Raw binary read order: Z,Y,X; NumPy reshape: (Z,Y,X)\n"
            "PyTorch tensor order: (B,C,D,H,W) = (B,1,Z,Y,X)\n"
            "No transpose is applied after raw reshape.\n"
            "Flow channels: ux,uy,uz corresponding to physical X,Y,Z axes.\n"
        )

    # ----------------------------------------------------------------------
    # Generate/load separate masks for p80 and p150. This avoids applying a
    # p80 heart mask to the independent p150 phantom.
    # ----------------------------------------------------------------------
    expected_shape = (
        cfg.keep_z_stop - cfg.keep_z_start,
        cfg.matrix_xyz[1],
        cfg.matrix_xyz[0],
    )

    if args.mask_dir:
        mask_root = Path(args.mask_dir)
        p80_mask_dir = mask_root / "p80"
        p150_mask_dir = mask_root / "p150"
    else:
        mask_root = output_dir / "phase_masks"
        p80_mask_dir = mask_root / "p80"
        p150_mask_dir = mask_root / "p150"

    if args.generate_masks or not p80_mask_dir.exists():
        print("Generating p80 phase 1..8 masks...")
        generate_phase_masks_and_save_i(
            data_dir=train_dir, out_dir=p80_mask_dir, phases=tuple(range(1,9)),
            frame=args.fr, matrix_xyz=cfg.matrix_xyz,
            keep_z=slice(cfg.keep_z_start, cfg.keep_z_stop), mask_threshold=0.2,
        )
    if args.generate_masks or not p150_mask_dir.exists():
        print("Generating p150 phase 1..8 masks...")
        generate_phase_masks_and_save_i(
            data_dir=test_dir, out_dir=p150_mask_dir, phases=tuple(range(1,9)),
            frame=args.fr, matrix_xyz=cfg.matrix_xyz,
            keep_z=slice(cfg.keep_z_start, cfg.keep_z_stop), mask_threshold=0.2,
        )

    p80_masks = load_phase_masks(p80_mask_dir, tuple(range(1,9)), expected_shape)
    p150_masks = load_phase_masks(p150_mask_dir, tuple(range(1,9)), expected_shape)

    # ----------------------------------------------------------------------
    # Cycle-level train/validation split.
    # ----------------------------------------------------------------------
    train_ds = XCATPETDataset(
        train_dir, phantom="p80", fr=args.fr, phases=tuple(range(1,33)),
        pair_within_phases=tuple(range(1,33)), phase_masks=p80_masks, cfg=cfg,
    )
    val_ds = XCATPETDataset(
        train_dir, phantom="p80", fr=args.fr, phases=tuple(range(33,41)),
        pair_within_phases=tuple(range(33,41)), phase_masks=p80_masks, cfg=cfg,
    )

    print(f"Training pairs:   {len(train_ds)}")
    print(f"Validation pairs: {len(val_ds)}")

    model = MotionCorrectionNet(
        base_channels=cfg.base_channels, attention_heads=cfg.attention_heads,
    ).to(device)

    explicit_ckpt = Path(args.checkpoint) if args.checkpoint else None
    training_dir = checkpoint_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    latest_path = training_dir / "latest_checkpoint.pt"

    if explicit_ckpt is not None:
        ckpt = torch.load(explicit_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded explicit checkpoint: {explicit_ckpt}")
    elif cfg.resume and latest_path.exists():
        print(f"Auto-resuming from: {latest_path}")

    if not args.skip_training:
        train_loader = make_loader(train_ds, cfg, shuffle=True)
        val_loader = make_loader(val_ds, cfg, shuffle=False)

        best = train_model(
            model, train_loader, val_loader, device, training_dir, cfg,
            resume_path=(latest_path if cfg.resume and explicit_ckpt is None else None),
            explicit_checkpoint=explicit_ckpt,
        )

        ckpt = torch.load(best, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best validation model for independent test: {best}")
    else:
        # For inference-only mode, prefer explicit checkpoint, then best, then latest.
        if explicit_ckpt is None:
            candidates = [training_dir / "best_model.pt", latest_path, training_dir / "final_model.pt"]
            chosen = next((q for q in candidates if q.exists()), None)
            if chosen is None:
                raise FileNotFoundError("No checkpoint available for --skip_training.")
            ckpt = torch.load(chosen, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded inference checkpoint: {chosen}")

    # ----------------------------------------------------------------------
    # Independent p150 adjacent-pair evaluation.
    # ----------------------------------------------------------------------
    test_out = inference_root / "adjacent_pairs"
    evaluate_test_pairs(
        model=model, data_dir=test_dir, phantom="p150", fr=args.fr,
        phase_masks=p150_masks, device=device, cfg=cfg, out_dir=test_out,
    )

    # ----------------------------------------------------------------------
    # Arbitrary-reference inference, e.g. ph26 -> ph16.
    # ----------------------------------------------------------------------
    if not args.skip_arbitrary_inference:
        arbitrary_out = inference_root / "arbitrary_pairs"
        for ref_phase, moving_phase in parse_pair_specs(args.arbitrary_pairs):
            print(f"Arbitrary inference: moving ph{moving_phase} -> reference ph{ref_phase}")
            run_arbitrary_pair_inference(
                model, test_dir, "p150", ref_phase, moving_phase, args.fr,
                p150_masks, device, cfg, arbitrary_out,
            )

    # ----------------------------------------------------------------------
    # Eight-gate downstream demonstration using p150-specific masks.
    # ----------------------------------------------------------------------
    gate_dir = inference_root / "gates"
    make_motion_corrected_gates(
        model=model, data_dir=test_dir, phantom="p150", fr=args.fr,
        phase_masks=p150_masks, device=device, cfg=cfg, out_dir=gate_dir,
    )
    save_gate_groups(inference_root / "gate_phase_groups.json")

    # ----------------------------------------------------------------------
    # Optional conventional registration for presentation only.
    # ----------------------------------------------------------------------
    if args.run_registration:
        reg_out = inference_root / "conventional_registration"
        for ref_phase, moving_phase in parse_pair_specs(args.registration_pairs):
            print(
                f"Conventional registration (QUALITATIVE ONLY): "
                f"ph{moving_phase} -> ph{ref_phase}"
            )
            run_conventional_registration_pair(
                test_dir, "p150", ref_phase, moving_phase, args.fr,
                p150_masks, cfg, reg_out,
            )

    print("\nCompleted.")
    print(f"Results:     {output_dir}")
    print(f"Inference:   {inference_root}")
    print(f"Checkpoints: {training_dir}")
    print(
        "\nIMPORTANT: conventional registration, when requested, is only a "
        "qualitative/presentation reference. It is NOT ground truth and is "
        "not used in training or EPE evaluation. True XCAT DVFs would require "
        "the original XCAT vector-generation output."
    )


if __name__ == "__main__":
    main()
