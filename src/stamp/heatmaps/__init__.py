#!/usr/bin/env python3

import logging
from collections.abc import Iterable
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Integer
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from PIL import Image
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.func import jacrev

from stamp.modeling.data import get_coords, get_stride
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.preprocessing import supported_extensions
from stamp.preprocessing.tiling import Microns, SlideMPP, TilePixels, get_slide_mpp_

_logger = logging.getLogger("stamp")

Image.MAX_IMAGE_PIXELS = None


# ------------------------------------------------------------------------
# Helper: load rejection thumbnail (same logic as extract_)
# ------------------------------------------------------------------------
def _get_rejection_thumb(
    slide: openslide.AbstractSlide,
    *,
    size: tuple[int, int],
    coords_um: np.ndarray,
    tile_size_um: Microns,
    default_slide_mpp: SlideMPP | None,
) -> Image.Image:
    inclusion_map = np.zeros(
        np.uint32(
            np.ceil(
                np.array(slide.dimensions)
                * get_slide_mpp_(slide, default_mpp=default_slide_mpp)
                / tile_size_um
            )
        ),
        dtype=bool,
    )

    for y, x in np.floor(coords_um / tile_size_um).astype(np.uint32):
        inclusion_map[y, x] = True

    thumb = slide.get_thumbnail(size).convert("RGBA")

    # Create RGBA image with transparent background
    # Where inclusion_map is True → keep thumbnail pixel (alpha=255)
    # Where False → make it fully transparent (alpha=0)
    mask_array = np.where(
        inclusion_map.transpose()[:, :, None],
        [255, 255, 255, 255],  # keep pixel
        [0, 0, 0, 0],  # transparent
    ).astype(np.uint8)

    # Resize mask to match thumbnail size
    mask_im = Image.fromarray(mask_array).resize(
        thumb.size, resample=Image.Resampling.NEAREST
    )

    # Apply mask to thumbnail (remove background)
    thumb = Image.composite(thumb, Image.new("RGBA", thumb.size, (0, 0, 0, 0)), mask_im)

    # ---- TIGHT CROP ----
    thumb_np = np.array(thumb)
    alpha_channel = thumb_np[:, :, 3]
    nonzero = np.argwhere(alpha_channel > 0)

    if nonzero.size > 0:
        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0) + 1
        thumb = thumb.crop((x_min, y_min, x_max, y_max))
    else:
        # Return 1x1 transparent image if empty
        thumb = Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    # ---------------------------------------
    # Remove white background (RGB > 230)
    # ---------------------------------------
    thumb_arr = np.array(thumb)

    # Identify pixels where R, G, and B are all greater than threshold (i.e. white)
    bg_threshold = 230
    is_white = np.all(thumb_arr[:, :, :3] > bg_threshold, axis=2)

    # Make those white pixels transparent
    thumb_arr[is_white, 3] = 0

    # Convert back to RGBA image
    thumb = Image.fromarray(thumb_arr, mode="RGBA")

    return thumb


# ------------------------------------------------------------------------
# GradCAM computation
# ------------------------------------------------------------------------


def _gradcam_per_category(
    model: VisionTransformer,
    feats: Float[Tensor, "tile feat"],
    coords: Float[Tensor, "tile 2"],
) -> Float[Tensor, "tile category"]:
    feat = -1  # feats dimension

    return (
        (
            feats
            * jacrev(
                lambda bags: model.forward(
                    bags=bags.unsqueeze(0),
                    coords=coords.unsqueeze(0),
                    mask=None,
                ).squeeze(0)
            )(feats)
        )
        .mean(feat)  # type: ignore
        .abs()
    ).permute(-1, -2)


# ------------------------------------------------------------------------
# Convert tile scores to grid image
# ------------------------------------------------------------------------
def _vals_to_im(
    scores: Float[Tensor, "tile feat"],
    coords_norm: Integer[Tensor, "tile coord"],
    grid_shape: tuple[int, int],
) -> Float[Tensor, "height width category"]:
    im = torch.zeros(
        (grid_shape[0], grid_shape[1], scores.shape[1]),
        dtype=scores.dtype,
        device=scores.device,
    )
    for idx, (y, x) in enumerate(coords_norm):
        im[y, x] = scores[idx]
    return im


# ------------------------------------------------------------------------
# Plot a small GradCAM heatmap figure
# ------------------------------------------------------------------------
def _save_gradcam_figure(
    cam_array: np.ndarray,
    out_path: Path,
    title: str,
    cmap_name: str = "plasma",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cam_array, cmap=cmap_name)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    _logger.info(f"Saved GradCAM figure: {out_path}")


# ------------------------------------------------------------------------
# Main heatmaps function
# ------------------------------------------------------------------------
@torch.no_grad()
def heatmaps_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    topk: int,
    bottomk: int,
) -> None:
    model = LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()

    if not hasattr(model.vision_transformer, "categories"):
        model.vision_transformer.categories = ["regression_target"]

    slides = (
        (wsi_dir / slide for slide in slide_paths)
        if slide_paths is not None
        else (p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}"))
    )

    for wsi_path in slides:
        h5_path = feature_dir / wsi_path.with_suffix(".h5").name
        if not h5_path.exists():
            _logger.warning(f"No feature file found for {wsi_path}. Skipping.")
            continue

        slide_output_dir = output_dir / wsi_path.stem
        slide_output_dir.mkdir(exist_ok=True, parents=True)

        slide = openslide.open_slide(wsi_path)
        slide_mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
        assert slide_mpp is not None

        with h5py.File(h5_path) as h5:
            feats = torch.tensor(h5["feats"][:]).float().to(device)
            coords_info = get_coords(h5)
            coords_um = torch.tensor(
                coords_info.coords_um, dtype=torch.float32, device=device
            )
            stride_um = Microns(get_stride(coords_um.cpu()))
            tile_size_slide_px = TilePixels(
                int(round(float(coords_info.tile_size_um) / slide_mpp))
            )

        coords_norm = torch.floor(coords_um / stride_um).long()
        grid_h = coords_norm[:, 1].max().item() + 1
        grid_w = coords_norm[:, 0].max().item() + 1
        grid_shape = (grid_h, grid_w)

        # ----------- GradCAM computation ----------
        gradcam = _gradcam_per_category(
            model=model.vision_transformer,
            feats=feats,
            coords=coords_um,
        )

        gradcam_2d = (
            _vals_to_im(
                gradcam,
                coords_norm[:, [1, 0]],
                grid_shape,
            )
            .cpu()
            .numpy()[..., 0]
        )

        gradcam_2d_up = np.repeat(gradcam_2d, 20, axis=0)
        gradcam_2d_up = np.repeat(gradcam_2d_up, 20, axis=1)

        ### RECORTE TIGHT ###
        nonzero = np.argwhere(gradcam_2d_up > 0)
        if nonzero.size > 0:
            y_min, x_min = nonzero.min(axis=0)
            y_max, x_max = nonzero.max(axis=0) + 1
            gradcam_2d_up = gradcam_2d_up[y_min:y_max, x_min:x_max]
        else:
            gradcam_2d_up = np.zeros((1, 1), dtype=np.float32)

        gradcam_norm = (gradcam_2d_up - gradcam_2d_up.min()) / (
            gradcam_2d_up.max() - gradcam_2d_up.min() + 1e-8
        )

        alpha_mask = (gradcam_norm > 0).astype(float)

        gradcam_cmap = plt.get_cmap("plasma")
        gradcam_rgba = gradcam_cmap(gradcam_norm)
        gradcam_rgba[..., 3] = alpha_mask

        gradcam_arr = (gradcam_rgba * 255).astype(np.uint8)
        gradcam_img = Image.fromarray(gradcam_arr, mode="RGBA")

        gradcam_path = slide_output_dir / f"{wsi_path.stem}_gradcam.png"
        gradcam_img.save(gradcam_path)
        _logger.info(f"Saved GradCAM RGBA to {gradcam_path}")

        # ----------- Per-tile prediction ----------
        scores = []
        for i in range(len(feats)):
            out = model.vision_transformer(
                bags=feats[i : i + 1].unsqueeze(0),
                coords=coords_um[i : i + 1].unsqueeze(0),
                mask=torch.zeros(1, 1, dtype=torch.bool, device=device),
            ).squeeze(0)
            scores.append(out)
        scores = torch.stack(scores).view(-1, 1)

        scores_2d = (
            _vals_to_im(
                scores,
                coords_norm[:, [1, 0]],
                grid_shape,
            )
            .cpu()
            .numpy()[..., 0]
        )

        scores_2d_up = np.repeat(scores_2d, 20, axis=0)
        scores_2d_up = np.repeat(scores_2d_up, 20, axis=1)

        ### RECORTE TIGHT ###
        nonzero = np.argwhere(scores_2d_up > 0)
        if nonzero.size > 0:
            y_min, x_min = nonzero.min(axis=0)
            y_max, x_max = nonzero.max(axis=0) + 1
            scores_2d_up = scores_2d_up[y_min:y_max, x_min:x_max]
        else:
            scores_2d_up = np.zeros((1, 1), dtype=np.float32)

        # Optional: inverse transform scores if a scaler is available
        scaler_path = checkpoint_path.parent / "scaler.joblib"
        if scaler_path.exists():
            from sklearn.externals import (
                joblib,
            )  # or just `import joblib` if you're using it directly

            scaler = joblib.load(scaler_path)
            flat_scores = scores_2d_up.flatten().reshape(-1, 1)
            scores_2d_up = scaler.inverse_transform(flat_scores).reshape(
                scores_2d_up.shape
            )

        # Normalize with fixed min=0
        scores_norm = scores_2d_up / (scores_2d_up.max() + 1e-8)

        alpha_mask_pred = (scores_norm > 0).astype(float)

        cmap_pred = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
        pred_rgba = cmap_pred(scores_norm)
        pred_rgba[..., 3] = alpha_mask_pred

        pred_arr = (pred_rgba * 255).astype(np.uint8)
        pred_img = Image.fromarray(pred_arr, mode="RGBA")

        pred_path = slide_output_dir / f"{wsi_path.stem}_prediction.png"
        pred_img.save(pred_path)
        _logger.info(f"Saved prediction RGBA to {pred_path}")

        # ----------- Thumbnail generation ----------
        thumb = _get_rejection_thumb(
            slide,
            size=(pred_arr.shape[1], pred_arr.shape[0]),
            coords_um=coords_um.cpu().numpy(),
            tile_size_um=stride_um,
            default_slide_mpp=default_slide_mpp,
        )
        thumb_path = slide_output_dir / f"{wsi_path.stem}_thumb.png"
        thumb.save(thumb_path)
        _logger.info(f"Saved thumbnail to {thumb_path}")

        # ----------- Overlay ----------
        heat_img = pred_img.resize(thumb.size, Image.Resampling.NEAREST)

        alpha_factor = 0.6
        heat_arr = np.array(heat_img).astype(np.float32)
        heat_arr[..., 3] *= alpha_factor
        heat_arr = np.clip(heat_arr, 0, 255).astype(np.uint8)
        heat_img = Image.fromarray(heat_arr, mode="RGBA")

        overlay = Image.alpha_composite(
            Image.new("RGBA", thumb.size, (255, 255, 255, 0)),
            Image.alpha_composite(thumb.convert("RGBA"), heat_img),
        )
        overlay_path = slide_output_dir / f"{wsi_path.stem}_overlay.png"
        overlay.save(overlay_path)
        _logger.info(f"Saved overlay image to {overlay_path}")

        # ----------- Overview Figure (Matplotlib style) -----------
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)

        # --- THUMBNAIL (fila 0, ocupa toda la fila) ---
        ax_thumb = fig.add_subplot(gs[0, 0])
        ax_thumb.imshow(thumb)
        ax_thumb.set_title("Thumbnail")
        ax_thumb.axis("off")

        # --- GRADCAM (fila 1) ---
        ax_gradcam = fig.add_subplot(gs[1, 0])
        im_gradcam = ax_gradcam.imshow(gradcam_norm, cmap="plasma", alpha=alpha_mask)
        ax_gradcam.set_title("Grad-CAM")
        ax_gradcam.axis("off")
        cbar0 = fig.colorbar(
            im_gradcam,
            ax=ax_gradcam,
            orientation="horizontal",
            fraction=0.046,
            pad=0.04,
        )
        cbar0.set_label("Attention Score")

        # --- HEATMAP (fila 2) ---
        cmap_pred = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
        vmin = scores_2d_up.min() if scaler_path.exists() else 0.0
        vmax = scores_2d_up.max() if scaler_path.exists() else 1.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        ax_pred = fig.add_subplot(gs[2, 0])
        im_pred = ax_pred.imshow(
            scores_2d_up, cmap=cmap_pred, norm=norm, alpha=alpha_mask_pred
        )
        ax_pred.set_title("Prediction Heatmap")
        ax_pred.axis("off")
        cbar1 = fig.colorbar(
            im_pred, ax=ax_pred, orientation="horizontal", fraction=0.046, pad=0.04
        )
        cbar1.set_label(f"Prediction Score ({vmin:.2f}–{vmax:.2f})")

        # --- Save figure ---
        overview_path = slide_output_dir / f"{wsi_path.stem}_overview.png"
        fig.savefig(overview_path, bbox_inches="tight")
        plt.close(fig)
        _logger.info(f"Saved matplotlib overview to {overview_path}")

        # ----------- Export top and bottom tiles ----------
        coords_np = coords_um.cpu().numpy()
        scores_np = scores.cpu().numpy().flatten()

        # Combine coordinates and scores
        coords_scores = list(zip(coords_np.tolist(), scores_np.tolist()))

        # Sort by score (descending)
        coords_scores_sorted = sorted(coords_scores, key=lambda x: x[1], reverse=True)

        # Create subdirectories
        top_dir = slide_output_dir / "top_tiles"
        bottom_dir = slide_output_dir / "bottom_tiles"
        top_dir.mkdir(parents=True, exist_ok=True)
        bottom_dir.mkdir(parents=True, exist_ok=True)

        # Export top-k tiles
        for i, (coord, score) in enumerate(coords_scores_sorted[:topk]):
            tile_px = np.floor(np.array(coord) / stride_um).astype(int)
            region = slide.read_region(
                location=tuple((tile_px * tile_size_slide_px).tolist()),
                level=0,
                size=(tile_size_slide_px,) * 2,
            ).convert("RGB")
            region.save(top_dir / f"{wsi_path.stem}_top{i + 1}_{score:.3f}.png")
            _logger.info(f"Saved top-{i + 1} tile to {top_dir}")

        # Export bottom-k tiles
        for i, (coord, score) in enumerate(coords_scores_sorted[-bottomk:]):
            tile_px = np.floor(np.array(coord) / stride_um).astype(int)
            region = slide.read_region(
                location=tuple((tile_px * tile_size_slide_px).tolist()),
                level=0,
                size=(tile_size_slide_px,) * 2,
            ).convert("RGB")
            region.save(bottom_dir / f"{wsi_path.stem}_bottom{i + 1}_{score:.3f}.png")
            _logger.info(f"Saved bottom-{i + 1} tile to {bottom_dir}")
