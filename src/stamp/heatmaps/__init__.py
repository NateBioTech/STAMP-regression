import logging
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast, no_type_check

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Integer
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from PIL import Image
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.func import jacrev  # pyright: ignore[reportPrivateImportUsage]

from stamp.modeling.data import get_coords, get_stride
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.preprocessing import supported_extensions
from stamp.preprocessing.tiling import Microns, SlideMPP, TilePixels, get_slide_mpp_

_logger = logging.getLogger("stamp")


def _gradcam_per_category(
    model: VisionTransformer,
    feats: Float[Tensor, "tile feat"],
    coords: Float[Tensor, "tile 2"],
) -> Float[Tensor, "tile category"]:  # returns [tile, 1]

    # Forward pass  fills SelfAttention.attn_weights in each layer
    _ = model(
        bags=feats.unsqueeze(0),
        coords=coords.unsqueeze(0),
        mask=torch.zeros(1, len(feats), dtype=torch.bool, device=feats.device),
    )

    # Compute Attention Rollout
    attn_rollout = None
    for layer in model.transformer.layers:
        attn = layer[0].attn_weights  # SelfAttention.attn_weights
        if attn is None:
            raise RuntimeError("Attention weights not stored in SelfAttention layer.")

        attn = attn.mean(0)  # Average over heads  (seq, seq)
        attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows

        if attn_rollout is None:
            attn_rollout = attn
        else:
            attn_rollout = attn_rollout @ attn

    # Extract CLS attention to tiles
    cls_attn = attn_rollout[0, 1:]  # Attention from CLS token to all tiles

    # Normalize to [0,1]
    cam = cls_attn.cpu().unsqueeze(1)  # [tile, 1]  to match your current pipeline
    cam -= cam.min()
    cam /= cam.max().clamp(min=1e-8)

    return cam



def _vals_to_im(
    scores: Float[Tensor, "tile feat"],
    coords_norm: Integer[Tensor, "tile coord"],
) -> Float[Tensor, "width height category"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = coords_norm.max(0).values.flip(0) + 1
    im = torch.zeros((*size.tolist(), *scores.shape[1:])).type_as(scores)

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def _show_thumb(
    slide, thumb_ax: Axes, attention: Tensor, default_slide_mpp: SlideMPP | None
) -> np.ndarray:
    mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


@no_type_check
def _show_class_map(
    class_ax: Axes,
    top_score_indices: Integer[Tensor, "width height"],
    gradcam_2d: Float[Tensor, "width height category"],
    categories: Collection[str],
    fig=None,
) -> None:
    class_ax.axis("off")

    if len(categories) == 1:
        # Regression case
        cam = gradcam_2d[..., 0].cpu().numpy()
        alpha_mask = (cam != 0).astype(float)
        im = class_ax.imshow(cam, cmap="plasma", alpha=alpha_mask)

        if fig is not None:
            cbar = fig.colorbar(im, ax=class_ax,
                                 orientation="horizontal",
                                 fraction=0.046, pad=0.04)
            cbar.set_label("Grad–CAM score")

        class_ax.set_title(f"{categories[0]} (Grad–CAM map)")

    else:
        # Classification case
        cmap = plt.get_cmap("Pastel1")
        classes = cast(np.ndarray, cmap(top_score_indices.cpu().numpy()))
        classes[..., -1] = (gradcam_2d.sum(-1) > 0).cpu().numpy().astype(float)
        im = class_ax.imshow(classes)

        if fig is not None:
            cbar = fig.colorbar(im, ax=class_ax,
                                 orientation="horizontal",
                                 fraction=0.046, pad=0.04)
            cbar.set_label("Class presence")

        class_ax.legend(
            handles=[Patch(facecolor=cmap(i), label=cat)
                     for i, cat in enumerate(categories)],
            loc="lower center",
            ncol=len(categories),
            bbox_to_anchor=(0.5, -0.2),
        )
        class_ax.set_title("Class activation map")



@torch.no_grad()
def per_tile_prediction(
    model: VisionTransformer,
    feats: Float[Tensor, "tile feat"],
    coords: Float[Tensor, "tile 2"],
) -> Float[Tensor, "tile 1"]:
    outs = []
    for i in range(len(feats)):
        out = model.forward(
            bags=feats[i : i + 1].unsqueeze(0),   
            coords=coords[i : i + 1].unsqueeze(0),
            mask=torch.zeros(1, 1, dtype=torch.bool, device=feats.device),
        ).squeeze(0)                              
        outs.append(out)
    return torch.stack(outs)[:, None]             



def heatmaps_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    # top tiles
    topk: int,
    bottomk: int,
) -> None:
    model = LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()

    if not hasattr(model, "categories"):
        model.categories = ["regression_target"]

    # Collect slides to generate heatmaps for
    if slide_paths is not None:
        wsis_to_process = (wsi_dir / slide for slide in slide_paths)
    else:
        wsis_to_process = (
            p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}")
        )

    for wsi_path in wsis_to_process:
        h5_path = feature_dir / wsi_path.with_suffix(".h5").name

        if not h5_path.exists():
            _logger.info(f"could not find matching h5 file at {h5_path}. Skipping...")
            continue

        slide_output_dir = output_dir / h5_path.stem
        slide_output_dir.mkdir(exist_ok=True, parents=True)
        _logger.info(f"creating heatmaps for {wsi_path.name}")

        slide = openslide.open_slide(wsi_path)
        slide_mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
        assert slide_mpp is not None, "could not determine slide MPP"

        with h5py.File(h5_path) as h5:
            feats = (
                torch.tensor(
                    h5["feats"][:]  # pyright: ignore[reportIndexIssue]
                )
                .float()
                .to(device)
            )
            coords_info = get_coords(h5)
            coords_um = coords_info.coords_um
            stride_um = Microns(get_stride(coords_um))

            tile_size_slide_px = TilePixels(
                int(round(float(coords_info.tile_size_um) / slide_mpp))
            )

        # grid coordinates, i.e. the top-left most tile is (0, 0), the one to its right (0, 1) etc.
        coords_norm = (coords_um / stride_um).round().long()

        # coordinates as used by OpenSlide
        coords_tile_slide_px = torch.round(coords_um / slide_mpp).long()

        # Score for the entire slide
        slide_score = (
            model.vision_transformer(
                bags=feats.unsqueeze(0),
                coords=coords_um.unsqueeze(0),
                mask=torch.zeros(1, len(feats), dtype=torch.bool, device=device),
            )
            .squeeze(0)
        )

        gradcam = _gradcam_per_category(
            model=model.vision_transformer,
            feats=feats,
            coords=coords_um,
        )  # shape: [tile, category]
        gradcam_2d = _vals_to_im(
            gradcam,
            coords_norm,
        ).detach()  # shape: [width, height, category]


        """
        scores = torch.softmax(
            model.vision_transformer.forward(
                bags=feats.unsqueeze(-2),
                coords=coords_um.unsqueeze(-2),
                mask=torch.zeros(len(feats), 1, dtype=torch.bool, device=device),
            ),
            dim=1,
        )  # shape: [tile, category]
        """

        scores = per_tile_prediction(
            model=model.vision_transformer,
            feats=feats,
            coords=coords_um,
        )  # shape [tile, 1]

        scores_2d = _vals_to_im(scores.squeeze(-1), coords_norm).detach()   # [H,W,1]
        im_data   = scores_2d.squeeze(-1).cpu().numpy()                     # 2-D
        alpha_mask = (im_data != 0).astype(float)                           # 2-D


        fig, axs = plt.subplots(
            nrows=2, ncols=max(2, len(model.categories)), figsize=(12, 8)
        )

        # Thumbnail en [0,0]
        thumb = _show_thumb(
            slide=slide,
            thumb_ax=axs[0, 0],
            attention=scores_2d.squeeze(-1),
            default_slide_mpp=default_slide_mpp,
        )

        # Grad–CAM (o clasificación) en [0,1:]
        _show_class_map(
            class_ax=axs[0, 1],
            top_score_indices=torch.zeros(scores_2d.shape[:2], dtype=torch.long, device=scores_2d.device),
            gradcam_2d=gradcam_2d,
            categories=model.categories,
            fig=fig,
        )

        from matplotlib.colors import Normalize

        # Define once, before the loop
        norm = Normalize(vmin=0.0, vmax=1.0, clip=True)

        # Prediction maps and saving in [1, *]
        from matplotlib.colors import LinearSegmentedColormap

        blue_red = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

        for pos_idx, category in enumerate(model.categories):
            ax = axs[1, pos_idx]
            im_data = scores_2d.squeeze(-1).cpu().numpy()
            alpha_mask = (im_data != 0).astype(float)

            # REPLACE this block
            # rgba_f = blue_red(im_data)
            # rgba_f[..., 3] = alpha_mask
            # im = ax.imshow(rgba_f)
            #
            # WITH this:
            im = ax.imshow(
                im_data,
                cmap=blue_red,
                norm=norm,        # clips <0 to blue and >1 to red
                alpha=alpha_mask  # transparency mask
            )
            # 

            # 2) Add horizontal colorbar
            cbar = fig.colorbar(
                im, ax=ax,
                orientation="horizontal",
                fraction=0.046, pad=0.04
            )
            cbar.set_label("Prediction (0–1)")

            # Rename the title
            ax.set_title(f"Prediction map: {category} ({slide_score[pos_idx]:.2f})")
            ax.axis("off")

            # Save the image exactly as shown
            #out_path = slide_output_dir / f"{h5_path.stem}-{category}={slide_score[pos_idx]:.2f}.png"
            #arr = (im.cmap(im.norm(im_data)) * 255).astype(np.uint8)
            #Image.fromarray(arr).save(out_path)


            # Save the image with alpha mask
            out_path = slide_output_dir / f"{h5_path.stem}-{category}={slide_score[pos_idx]:.2f}.png"
            # Get the normalized RGBA float array
            rgba = im.cmap(im.norm(im_data))           # shape H×W×4
            # Overwrite its alpha channel with your mask
            rgba[..., 3] = alpha_mask                  # 1.0 where tile, 0.0 elsewhere
            # Convert to 0-255 uint8
            out_arr = (rgba * 255).astype(np.uint8)
            # Save as RGBA
            Image.fromarray(out_arr, mode="RGBA").save(out_path)

            _logger.info(f"Saved heatmap to {out_path}")


            # Top-k y bottom-k
            category_score = scores[:, pos_idx].squeeze(-1)
            for score, tile_idx in zip(*category_score.topk(topk)):
                img = slide.read_region(
                    tuple(coords_tile_slide_px[tile_idx].tolist()), 0,
                    (tile_size_slide_px, tile_size_slide_px)
                ).convert("RGB")
                img.save(slide_output_dir / f"top-{h5_path.stem}-{category}={score:.2f}.jpg")

            for score, tile_idx in zip(*(-category_score).topk(bottomk)):
                img = slide.read_region(
                    tuple(coords_tile_slide_px[tile_idx].tolist()), 0,
                    (tile_size_slide_px, tile_size_slide_px)
                ).convert("RGB")
                img.save(slide_output_dir / f"bottom-{h5_path.stem}-{category}={(-score):.2f}.jpg")


        for ax in axs.ravel():
            ax.axis("off")

        fig.savefig(
            slide_output_dir / f"overview-{h5_path.stem}.png"
        )
        plt.close(fig)

