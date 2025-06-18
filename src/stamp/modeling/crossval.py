import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import numpy as np
from lightning.pytorch.accelerators.accelerator import Accelerator
from pydantic import BaseModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Modified the original code
from stamp.modeling.data import (
    FeaturePath,
    GroundTruth,
    PandasLabel,
    PatientData,
    PatientId,
    filter_complete_patient_data_,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.deploy import _predict, _to_prediction_df
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.train import setup_model_for_training, train_model_
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.modeling.correlationplot import plot_correlation, plot_all_fold_correlations

from sklearn.metrics import r2_score, mean_squared_error   # NEW: functions for regression

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


class _Split(BaseModel):
    train_patients: set[PatientId]
    test_patients: set[PatientId]


class _Splits(BaseModel):
    splits: Sequence[_Split]


def regression_crossval_(
    clini_table: Path,
    slide_table: Path,
    feature_dir: Path,
    output_dir: Path,
    patient_label: PandasLabel,
    regression_label: PandasLabel,
    filename_label: PandasLabel,
    n_splits: int,
    # Dataset and -loader parameters
    bag_size: int,
    num_workers: int,
    # Training parameters
    batch_size: int,
    max_epochs: int,
    patience: int,
    accelerator: str | Accelerator,
    # Experimental features
    use_vary_precision_transform: bool,
    use_alibi: bool,
) -> None:
    patient_to_ground_truth: Final[dict[PatientId, GroundTruth]] = (
        patient_to_ground_truth_from_clini_table_(
            clini_table_path=clini_table,
            ground_truth_label=regression_label,
            patient_label=patient_label,
        )
    )

    slide_to_patient: Final[dict[FeaturePath, PatientId]] = (
        slide_to_patient_from_slide_table_(
            slide_table_path=slide_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            filename_label=filename_label,
        )
    )

    # Clean data (remove slides without ground truth, missing features, etc.)
    patient_to_data: Final[Mapping[PatientId, PatientData]] = filter_complete_patient_data_(
        patient_to_ground_truth=patient_to_ground_truth,
        slide_to_patient=slide_to_patient,
        drop_patients_with_missing_ground_truth=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_file = output_dir / "splits.json"

    # Generate the splits, or load them from the splits file if they already exist
    if not splits_file.exists():
        splits = _get_splits(patient_to_data=patient_to_data, n_splits=n_splits)
        with open(splits_file, "w") as fp:
            fp.write(splits.model_dump_json(indent=4))
    else:
        _logger.debug(f"reading splits from {splits_file}")
        with open(splits_file, "r") as fp:
            splits = _Splits.model_validate_json(fp.read())

    patients_in_splits = {
        patient
        for split in splits.splits
        for patient in [*split.train_patients, *split.test_patients]
    }

    if patients_without_ground_truth := patients_in_splits - patient_to_data.keys():
        raise RuntimeError(
            "The splits file contains some patients we don't have information for in the clini / slide table: "
            f"{patients_without_ground_truth}"
        )

    if ground_truths_not_in_split := patient_to_data.keys() - patients_in_splits:
        _logger.warning(
            "Some of the entries in the clini / slide table are not in the crossval split: "
            f"{ground_truths_not_in_split}"
        )

    """ # Not used for regression
    categories = categories or sorted(
        {
            patient_data.ground_truth
            for patient_data in patient_to_data.values()
            if patient_data.ground_truth is not None
        }
    )
    """

    for split_i, split in enumerate(splits.splits):
        split_dir = output_dir / f"split-{split_i}"

        # Fit MinMaxScaler on training set only
        train_values = [
            patient_to_ground_truth[pid]
            for pid in split.train_patients
            if patient_to_ground_truth[pid] is not None
        ]

        scaler = MinMaxScaler(feature_range=(0, 1)).fit(
            np.array(train_values).reshape(-1, 1)
        )

        split_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, split_dir / "scaler.pkl")

        # Create a scaled copy of ground truth values (avoids cross-fold contamination)
        scaled_gt = patient_to_ground_truth.copy()
        for pid in split.train_patients | split.test_patients:
            if pid in scaled_gt and scaled_gt[pid] is not None:
                _val = scaler.transform([[scaled_gt[pid]]])[0][0]
                scaled_gt[pid] = float(np.clip(_val, 0, 1))

        # Build fold-specific PatientData using the scaled ground truths
        patient_to_data_fold: Mapping[PatientId, PatientData] = filter_complete_patient_data_(
            patient_to_ground_truth=scaled_gt,
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=True,
        )

        if (split_dir / "patient-preds.csv").exists():
            _logger.info(
                f"skipping training for split {split_i}, "
                "as a model checkpoint is already present"
            )
            continue

        # Train the model on training patients
        if not (split_dir / "model.ckpt").exists():
            model, train_dl, valid_dl = setup_model_for_training(
                clini_table=clini_table,
                slide_table=slide_table,
                feature_dir=feature_dir,
                ground_truth_label=regression_label,
                bag_size=bag_size,
                num_workers=num_workers,
                batch_size=batch_size,
                # Use fold-specific patient_to_data with scaled regression targets
                # (unlike classification, where categories are discrete and unchanged)
                patient_to_data={
                    pid: pd
                    for pid, pd in patient_to_data_fold.items()
                    if pid in split.train_patients
                },
                train_transform=(
                    VaryPrecisionTransform(min_fraction_bits=1)
                    if use_vary_precision_transform
                    else None
                ),
                use_alibi=use_alibi,
            )

            model = train_model_(
                output_dir=split_dir,
                model=model,
                train_dl=train_dl,
                valid_dl=valid_dl,
                max_epochs=max_epochs,
                patience=patience,
                accelerator=accelerator,
            )
        else:
            model = LitVisionTransformer.load_from_checkpoint(split_dir / "model.ckpt")

        # Deploy model on test patients and generate predictions
        if not (split_dir / "patient-preds.csv").exists():
            predictions = _predict(
                model=model,
                # Use fold-specific test data with scaled regression targets
                # (in classification, data does not require per-fold transformation)
                patient_to_data={
                    pid: pd
                    for pid, pd in patient_to_data_fold.items()
                    if pid in split.test_patients
                },
                num_workers=num_workers,
                accelerator=accelerator,
            )

            df = _to_prediction_df(
                # For regression, we use scaled ground truth values (required by training)
                patient_to_reg_value=scaled_gt,
                predictions=predictions,
                patient_label=patient_label,
                ground_truth_label=regression_label,
            )

            # Inverse transform predictions and compute regression metrics
            y_true = df[regression_label].values.reshape(-1, 1)
            y_pred = df["pred"].values.reshape(-1, 1)

            # Inverse MinMaxScaler to return predictions to original scale
            y_true_descaled = scaler.inverse_transform(y_true).flatten()
            y_pred_descaled = scaler.inverse_transform(y_pred).flatten()

            df["true_descaled"] = y_true_descaled
            df["pred_descaled"] = y_pred_descaled
            df["error_descaled"] = np.abs(y_pred_descaled - y_true_descaled)

            # Compute basic regression metrics
            r2 = r2_score(y_true_descaled, y_pred_descaled)
            rmse = mean_squared_error(y_true_descaled, y_pred_descaled)
            

            df.attrs["R2"] = r2
            df.attrs["RMSE"] = rmse

            df.to_csv(split_dir / "patient-preds.csv", index=False)

            plot_correlation(
                y_true=y_true_descaled,
                y_pred=y_pred_descaled,
                output_dir=split_dir,
            )

        # Plot training vs validation loss curve
        history_csv = split_dir / "history.csv"
        if history_csv.exists():
            history_df = pd.read_csv(history_csv)

            plt.figure(figsize=(8, 5))
            plt.plot(history_df["epoch"], history_df["training_loss"], label="Training Loss")
            plt.plot(history_df["epoch"], history_df["validation_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Fold {split_i} Training vs Validation Loss")
            plt.legend()
            plt.grid(True)

            loss_plot_path = split_dir / "validation_plot.png"
            plt.savefig(loss_plot_path, dpi=300)
            plt.close()

            print(f"Validation loss plot saved at {loss_plot_path}")

    plot_all_fold_correlations(output_dir=output_dir, n_splits=n_splits)


def _get_splits(
    *, patient_to_data: Mapping[PatientId, PatientData[Any]], n_splits: int
) -> _Splits:
    """
    Create stratified cross-validation splits for regression by binning continuous targets.

    Args:
        patient_to_data (Mapping[PatientId, PatientData]): Dictionary mapping patient IDs
            to their corresponding data, which includes ground truth values.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        _Splits: A dataclass containing `n_splits` train/test patient splits.
    """
    patients = np.array(list(patient_to_data.keys()))
    y = np.array([patient_to_data[pid].ground_truth for pid in patients])

    bins = np.quantile(y, np.linspace(0, 1, 11))  # 10 quantile bins (deciles)
    y_binned = np.digitize(y, bins[1:-1])  # exclude the 0th and 100th percentiles

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    splits = _Splits(
        splits=[
            _Split(
                train_patients=set(patients[train_idx]),
                test_patients=set(patients[test_idx]),
            )
            for train_idx, test_idx in skf.split(patients, y_binned)
        ]
    )
    return splits