"""Helper classes to manage pytorch data."""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from itertools import groupby
from pathlib import Path
from typing import BinaryIO, Generic, NewType, TextIO, TypeAlias, TypeVar, cast

import h5py
import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Integer
from packaging.version import Version
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import stamp
from stamp.preprocessing.tiling import Microns, SlideMPP, TilePixels

_logger = logging.getLogger("stamp")


__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


PatientId: TypeAlias = str
GroundTruth: TypeAlias = float #str
FeaturePath = NewType("FeaturePath", Path)

Category: TypeAlias = str

# One instance
_Bag: TypeAlias = Float[Tensor, "tile feature"]
BagSize: TypeAlias = int
# The old classification line for _EncodedTarget:
# _EncodedTarget: TypeAlias = Bool[Tensor, "category_is_hot"]  # old classification approach
_EncodedTarget: TypeAlias = Float[Tensor, ""]  # new line for regression (single float per bag)
"""The ground truth, encoded numerically (currently: one-hot)"""
_Coordinates: TypeAlias = Float[Tensor, "tile 2"]

# A batch of the above
Bags: TypeAlias = Float[Tensor, "batch tile feature"]
BagSizes: TypeAlias = Integer[Tensor, "batch"]  # noqa: F821
# The old classification line for EncodedTargets:
# EncodedTargets: TypeAlias = Bool[Tensor, "batch category_is_hot"]  # old classification approach
EncodedTargets: TypeAlias = Float[Tensor, "batch"]  # new for regression (batch of floats)
CoordinatesBatch: TypeAlias = Float[Tensor, "batch tile 2"]

PandasLabel: TypeAlias = str

GroundTruthType = TypeVar("GroundTruthType", covariant=True)


@dataclass
class PatientData(Generic[GroundTruthType]):
    """All raw (i.e. non-generated) information we have on the patient."""

    _ = KW_ONLY
    ground_truth: GroundTruthType
    feature_files: Iterable[FeaturePath | BinaryIO]


def dataloader_from_patient_data(
    *,
    patient_data: Sequence[PatientData[GroundTruth | None]],
    bag_size: int | None,
    #categories: Sequence[Category] | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Callable[[Tensor], Tensor] | None,
) -> DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]]:
    
    """Creates a dataloader from patient data, encoding the ground truths.

    Args:
        patient_data: List of PatientData, each with features and a float label.
        bag_size: Number of tiles per bag. If None, use all available tiles.
        batch_size: Number of patients (bags) per batch.
        shuffle: Whether to shuffle patients during training.
        num_workers: Number of subprocesses for data loading.
        transform: Optional transform to apply to tile-level features.

    Returns:
        A PyTorch DataLoader yielding batches of:
        (Bags, Coordinates, BagSizes, RegressionTargets)
    """

    # CHANGED: instead of one-hot encoding, we parse float values directly 
    raw_ground_truths = []
    for patient in patient_data:
        # Convert ground truth to float (if missing, set to NaN)
        val = float(patient.ground_truth) if patient.ground_truth is not None else float("nan")
        raw_ground_truths.append(val)

    # NEW: store as float tensor 
    gt_tensor = torch.tensor(raw_ground_truths, dtype=torch.float32)

    # SAME: Build the dataset 
    ds = BagDataset(
        bags=[p.feature_files for p in patient_data],
        bag_size=bag_size,
        ground_truths=gt_tensor,       # <- regression: float vector instead of one-hot matrix
        transform=transform,
    )

    # CHANGED: Return just the DataLoader (no category list needed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_to_tuple,
    )


def _collate_to_tuple(
    items: list[tuple[_Bag, _Coordinates, BagSize, _EncodedTarget]],
) -> tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]:

    # CHANGED: Use padding since bags may have variable length (regression setting)
    bags = torch.nn.utils.rnn.pad_sequence(
        [bag for bag, _, _, _ in items],
        batch_first=True,
        padding_value=0.0
    )

    # SAME: Pad coordinates to align with padded bags
    coords = torch.nn.utils.rnn.pad_sequence(
        [coord for _, coord, _, _ in items],
        batch_first=True,
        padding_value=0.0
    )

    # SAME: Capture the actual number of tiles in each original bag
    bag_sizes = torch.tensor([bagsize for _, _, bagsize, _ in items])

    # CHANGED: For regression, each target is a float instead of a one-hot vector
    encoded_targets = torch.stack([
        et if torch.is_tensor(et) else torch.tensor(et, dtype=torch.float32)
        for _, _, _, et in items
    ])

    return bags, coords, bag_sizes, encoded_targets


@dataclass
class BagDataset(Dataset[tuple[_Bag, _Coordinates, BagSize, _EncodedTarget]]):
    """A dataset of bags of instances."""

    _: KW_ONLY
    bags: Sequence[Iterable[FeaturePath | BinaryIO]]
    """The `.h5` files containing the bags.

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x F,
    where N is the number of instances and F the number of features per instance.
    """

    bag_size: BagSize | None = None
    """The number of instances in each bag.

    For bags containing more instances,
    a random sample of `bag_size` instances will be drawn.
    Smaller bags are padded with zeros.
    If `bag_size` is None, all the samples will be used.
    """

    # CHANGED: For regression, the target is a float instead of one-hot
    ground_truths: Float[Tensor, "index"]

    transform: Callable[[Tensor], Tensor] | None

    def __post_init__(self) -> None:
        if len(self.bags) != len(self.ground_truths):
            raise ValueError(
                "the number of ground truths has to match the number of bags"
            )

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(
        self, index: int
    ) -> tuple[_Bag, _Coordinates, BagSize, _EncodedTarget]:
        # Collect all the features
        feats = []
        coords_um = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, "r") as h5:
                feats.append(
                    torch.from_numpy(h5["feats"][:])  # pyright: ignore[reportIndexIssue]
                )
                coords_um.append(get_coords(h5).coords_um)    # CHANGED: according to new coords accessor

        feats = torch.concat(feats).float()
        coords_um = torch.concat(coords_um).float()

        if self.transform is not None:
            feats = self.transform(feats)

        # Sample a subset, if required
        if self.bag_size is not None:
            return (
                *_to_fixed_size_bag(feats, coords=coords_um, bag_size=self.bag_size),
                self.ground_truths[index].reshape(()),  # CHANGED: shape [] for scalar regression
            )
        else:
            return (
                feats,
                coords_um,
                len(feats),
                self.ground_truths[index].reshape(()),  # CHANGED
            )


@dataclass
class CoordsInfo:
    coords_um: Tensor
    tile_size_um: Microns
    tile_size_px: TilePixels | None

    @property
    def mpp(self) -> SlideMPP:
        if not self.tile_size_px:
            raise RuntimeError(
                "tile size in pixels is not available. Please reextract them using `stamp preprocess`."
            )
        return SlideMPP(self.tile_size_um / self.tile_size_px)


def get_coords(feature_h5: h5py.File) -> CoordsInfo:
    coords = torch.from_numpy(feature_h5["coords"][:]).float()  # pyright: ignore[reportIndexIssue]
    coords_um: Tensor | None = None
    tile_size_um: Microns | None = None
    tile_size_px: TilePixels | None = None
    if (tile_size := feature_h5.attrs.get("tile_size", None)) and feature_h5.attrs.get(
        "unit", None
    ) == "um":
        # STAMP v2 format
        tile_size_um = Microns(float(tile_size))
        coords_um = coords
    elif tile_size := feature_h5.attrs.get("tile_size_um", None):
        # Newer STAMP format
        tile_size_um = Microns(float(tile_size))
        coords_um = coords
    elif round(feature_h5.attrs.get("tile_size", get_stride(coords))) == 224:
        # Historic STAMP format
        _logger.info(
            f"{feature_h5.filename}: tile stride is roughly 224, assuming coordinates have unit 256um/224px (historic STAMP format)"
        )
        tile_size_um = Microns(256.0)
        tile_size_px = TilePixels(224)
        coords_um = coords / 224 * 256

    if (version_str := feature_h5.attrs.get("stamp_version")) and (
        extraction_version := Version(version_str)
    ) > Version(stamp.__version__):
        raise RuntimeError(
            f"features were extracted with a newer version of stamp, please update your stamp to at least version {extraction_version}."
        )

    if not tile_size_px and "tile_size_px" in feature_h5.attrs:
        tile_size_px = TilePixels(int(feature_h5.attrs["tile_size_px"]))  # pyright: ignore[reportArgumentType]

    if not tile_size_um or coords_um is None:
        raise RuntimeError(
            "unable to infer coordinates from feature file. Please reextract them using `stamp preprocess`."
        )

    return CoordsInfo(coords_um, tile_size_um, tile_size_px)


def _to_fixed_size_bag(
    bag: _Bag, coords: _Coordinates, bag_size: BagSize
) -> tuple[_Bag, _Coordinates, BagSize]:
    """Samples a fixed-size bag of tiles from an arbitrary one.

    If the original bag did not have enough tiles,
    the bag is zero-padded to the right.
    """
    # get up to bag_size elements
    n_tiles, _dim_feats = bag.shape
    bag_idxs = torch.randperm(n_tiles)[:bag_size]
    bag_samples = bag[bag_idxs]
    coord_samples = coords[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded_bag = torch.cat(
        (
            bag_samples,
            torch.zeros(bag_size - bag_samples.shape[0], bag_samples.shape[1]),
        )
    )
    zero_padded_coord = torch.cat(
        (
            coord_samples,
            torch.zeros(bag_size - coord_samples.shape[0], coord_samples.shape[1]),
        )
    )
    return zero_padded_bag, zero_padded_coord, min(bag_size, len(bag))


def patient_to_ground_truth_from_clini_table_(
    *,
    clini_table_path: Path | TextIO,
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel,
) -> dict[PatientId, float]:
    """Loads the patients and their ground truths from a clini table.

    Returns:
        A mapping from patient ID to ground truth as float values.
    """

    clini_df = _read_table(
        clini_table_path,
        usecols=[patient_label, ground_truth_label],
        dtype=str,
    ).dropna()

    try:
        # CHANGED: convert ground truth values explicitly to float (for regression)
        patient_to_ground_truth: Mapping[PatientId, float] = {
            pid: float(gt)
            for pid, gt in clini_df.set_index(
                patient_label, verify_integrity=True
            )[ground_truth_label].items()
        }

    except KeyError as e:
        if patient_label not in clini_df:
            raise ValueError(
                f"{patient_label} was not found in clini table "
                f"(columns in clini table: {clini_df.columns})"
            ) from e
        elif ground_truth_label not in clini_df:
            raise ValueError(
                f"{ground_truth_label} was not found in clini table "
                f"(columns in clini table: {clini_df.columns})"
            ) from e
        else:
            raise e from e

    return dict(patient_to_ground_truth)



def slide_to_patient_from_slide_table_(
    *,
    slide_table_path: Path,
    feature_dir: Path,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
) -> dict[FeaturePath, PatientId]:
    """Creates a slide-to-patient mapping from a slide table."""
    slide_df = _read_table(
        slide_table_path,
        usecols=[patient_label, filename_label],
        dtype=str,
    )

    slide_to_patient: Mapping[FeaturePath, PatientId] = {
        FeaturePath(feature_dir / cast(str, k)): PatientId(cast(str, patient))
        for k, patient in slide_df.set_index(filename_label, verify_integrity=True)[
            patient_label
        ].items()
    }

    return slide_to_patient


def _read_table(path: Path | TextIO, **kwargs) -> pd.DataFrame:
    if not isinstance(path, Path):
        return pd.read_csv(path, **kwargs)
    elif path.suffix == ".xlsx":
        return pd.read_excel(path, **kwargs)
    elif path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    else:
        raise ValueError(
            "table to load has to either be an excel (`*.xlsx`) or csv (`*.csv`) file."
        )


def filter_complete_patient_data_(
    *,
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None],
    slide_to_patient: Mapping[FeaturePath, PatientId],
    drop_patients_with_missing_ground_truth: bool,
) -> Mapping[PatientId, PatientData]:
    """Aggregate information for all patients for which we have complete data.

    This will sort out slides with missing ground truth, missing features, etc.
    Patients with their ground truth set explicitly set to `None` will be _included_.

    Side effects:
        Checks feature paths' existance.
    """

    _log_patient_slide_feature_inconsistencies(
        patient_to_ground_truth=patient_to_ground_truth,
        slide_to_patient=slide_to_patient,
    )

    patient_to_slides: dict[PatientId, set[FeaturePath]] = {
        patient: set(slides)
        for patient, slides in groupby(
            slide_to_patient, lambda slide: slide_to_patient[slide]
        )
    }

    if not drop_patients_with_missing_ground_truth:
        patient_to_ground_truth = {
            **{patient_id: None for patient_id in patient_to_slides},
            **patient_to_ground_truth,
        }

    patients = {
        patient_id: PatientData(
            ground_truth=ground_truth, feature_files=existing_features_for_patient
        )
        for patient_id, ground_truth in patient_to_ground_truth.items()
        # Restrict to only patients which have slides and features
        if (slides := patient_to_slides.get(patient_id)) is not None
        and (
            existing_features_for_patient := {
                feature_path for feature_path in slides if feature_path.exists()
            }
        )
    }

    return patients


def _log_patient_slide_feature_inconsistencies(
    *,
    patient_to_ground_truth: Mapping[PatientId, GroundTruthType],
    slide_to_patient: Mapping[FeaturePath, PatientId],
) -> None:
    """Checks whether the arguments are consistent and logs all irregularities.

    Has no side effects outside of logging.
    """
    if (
        patients_without_slides := patient_to_ground_truth.keys()
        - slide_to_patient.values()
    ):
        _logger.warning(
            f"some patients have no associated slides: {patients_without_slides}"
        )

    if patients_without_ground_truth := (
        slide_to_patient.values() - patient_to_ground_truth.keys()
    ):
        _logger.warning(
            f"some patients have no clinical information: {patients_without_ground_truth}"
        )

    if slides_without_features := {
        slide for slide in slide_to_patient.keys() if not slide.exists()
    }:
        _logger.warning(
            f"some feature files could not be found: {slides_without_features}"
        )


def get_stride(coords: Float[Tensor, "tile 2"]) -> float:
    """Gets the minimum step width between any two coordintes."""
    xs: Tensor = coords[:, 0].unique(sorted=True)
    ys: Tensor = coords[:, 1].unique(sorted=True)
    stride = cast(
        float,
        min(
            (xs[1:] - xs[:-1]).min().item(),
            (ys[1:] - ys[:-1]).min().item(),
        ),
    )
    return stride
