# Modified the original code

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias, cast

import lightning
import numpy as np
import pandas as pd
import torch
import joblib
from lightning.pytorch.accelerators.accelerator import Accelerator

from stamp.modeling.data import (
    PandasLabel,
    PatientData,
    PatientId,
    dataloader_from_patient_data,
    filter_complete_patient_data_,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.correlationplot import plot_correlation

__all__ = ["deploy_regression_model_"]

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2024-2025 Marko van Treeck"
__license__ = "MIT"

_logger = logging.getLogger("stamp")

FloatValue: TypeAlias = float


def deploy_regression_model_(
    *,
    output_dir: Path,
    checkpoint_paths: Sequence[Path],
    clini_table: Path | None,
    slide_table: Path,
    feature_dir: Path,
    ground_truth_label: PandasLabel | None,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
    num_workers: int,
    accelerator: str | Accelerator,
) -> None:
    """
    Deploy a regression model (or an ensemble of models) by generating CSV predictions.

    A CSV file named "patient-preds-i.csv" is produced for each i-th checkpoint,
    plus an "patient-preds.csv" with the ensemble (mean) predictions.
    Columns: [<patient_label>, <ground_truth_label>, pred, error]
    """
    # Load each checkpoint as a LitVisionTransformer
    models = [
        LitVisionTransformer.load_from_checkpoint(
            str(ckpt_path)
            ).eval()
        for ckpt_path in checkpoint_paths
    ]
    if not models:
        raise ValueError("No checkpoints provided")

    # Check that all models share the same ground_truth_label
    ground_truth_labels = {m.ground_truth_label for m in models}
    if len(ground_truth_labels) != 1:
        raise RuntimeError(f"Inconsistent ground_truth_label among models: {ground_truth_labels}")
    model_ground_truth_label = models[0].ground_truth_label

    # If user specified ground_truth_label, verify or warn
    if ground_truth_label is not None and ground_truth_label != model_ground_truth_label:
        _logger.warning(
            f"Deployment ground_truth_label={ground_truth_label} does not match "
            f"the model's label={model_ground_truth_label}."
        )
    # Use the model's label if none was provided
    ground_truth_label = ground_truth_label or model_ground_truth_label

    # Build patient_data from the clini_table & slide_table
    output_dir.mkdir(exist_ok=True, parents=True)

    slide_to_patient = slide_to_patient_from_slide_table_(
        slide_table_path=slide_table,
        feature_dir=feature_dir,
        patient_label=patient_label,
        filename_label=filename_label,
    )
    
    patient_to_reg_value: Mapping[PatientId, float | None]

    if clini_table is not None:
        patient_to_reg_value = patient_to_ground_truth_from_clini_table_(
            clini_table_path=clini_table,
            ground_truth_label=ground_truth_label,
            patient_label=patient_label,
        )
    else:
        # If no table is provided, we do not know the ground truth; fill with None
        patient_to_reg_value = {pid: None for pid in set(slide_to_patient.values())}

    # Filter out patients with no existing features
    patient_to_data = filter_complete_patient_data_(
        patient_to_ground_truth=patient_to_reg_value,
        slide_to_patient=slide_to_patient,
        drop_patients_with_missing_ground_truth=False,
    )

    # Generate predictions from each checkpoint
    all_predictions_list = []
    for i, (model, ckpt_path) in enumerate(zip(models, checkpoint_paths)):
        preds_dict = _predict(
            model=model,
            patient_to_data=patient_to_data,
            num_workers=num_workers,
            accelerator=accelerator,
        )

        df = _to_prediction_df(
            patient_to_reg_value=patient_to_reg_value,
            predictions=preds_dict,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        )

        scaler_path = ckpt_path.parent / "scaler.pkl"
        try:
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                #df["pred"] = scaler.inverse_transform(df[["pred"]])

                if ground_truth_label in df.columns and df[ground_truth_label].notnull().any():
                    df["error"] = np.abs(df["pred"] - df[ground_truth_label])
                    y_true = df[ground_truth_label].values.astype(float)
                    y_pred = df["pred"].values.astype(float)
                    plot_correlation(
                        y_true,
                        y_pred,
                        output_dir / f"plot-{i}"  )      
                else:
                    df["error"] = None
            else:
                _logger.warning(f"No scaler found at {scaler_path}, skipping de-normalization")
                df["pred"] = df["pred"]
                df["error"] = None
        except Exception as e:
            _logger.warning(f"Error while rescaling predictions: {e}")
            df["pred"] = df["pred"]
            df["error"] = None

        csv_path = output_dir / f"patient-preds-{i}.csv"
        if df.empty:
            _logger.warning(f"DataFrame for split {i} is empty. Nothing was saved.")
        else:
            df.to_csv(csv_path, index=False)
            _logger.info(f"Predictions saved to: {csv_path}")

        all_predictions_list.append(preds_dict)

    if len(all_predictions_list) > 1:
        ensemble_preds = {}
        for pid in patient_to_data:
            arr = torch.stack([preds[pid] for preds in all_predictions_list], dim=0)
            mean_pred = arr.mean(dim=0)
            ensemble_preds[pid] = mean_pred
        ensemble_df = _to_prediction_df(
            patient_to_reg_value=patient_to_reg_value,
            predictions=ensemble_preds,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        )
        ensemble_df.to_csv(output_dir / "patient-preds.csv", index=False)
    else:
        single_preds_path = output_dir / "patient-preds-0.csv"
        final_path = output_dir / "patient-preds.csv"
        if single_preds_path.exists():
            import shutil
            shutil.copy(single_preds_path, final_path)
        else:
            _logger.warning("Did not find single checkpoint CSV to copy to patient-preds.csv")


def _predict(
    *,
    model: LitVisionTransformer,
    patient_to_data: Mapping[PatientId, PatientData[float | None]],
    num_workers: int,
    accelerator: str | Accelerator,
) -> dict[PatientId, torch.Tensor]:
    model.eval()
    torch.set_float32_matmul_precision("medium")

    """
    Return a dictionary {patient_id -> predicted_value} for each patient.
    """

    data_list = list(patient_to_data.values())

    dl = dataloader_from_patient_data(
        patient_data=data_list,
        bag_size=512,           # None
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        transform=None,
    )

    trainer = lightning.Trainer(accelerator=accelerator, 
                                devices=1, # Needs to be 1, otherwise half the predictions are missing for some reaso
                                logger=False)
    out_list = trainer.predict(model, dl)
    if not isinstance(out_list, list):
        raise RuntimeError("Expected list of prediction tensors from trainer.predict")

    preds_dict = {}
    for pat, out_tensor in zip(patient_to_data.keys(), out_list):
        preds_dict[pat] = out_tensor.squeeze(0)
    return preds_dict



def _to_prediction_df(
    *,
    patient_to_reg_value: Mapping[PatientId, float | None],
    predictions: Mapping[PatientId, torch.Tensor],
    patient_label: str,
    ground_truth_label: str,
    ) -> pd.DataFrame:
    """
    Convert predictions to a DataFrame with columns:
        [<patient_label>, <ground_truth_label>, pred, error]
    error is None if ground truth is missing.
    """
    rows = []
    for pid, pred_tensor in predictions.items():
        
        pred_value = float(pred_tensor.item()) # ie. 'PatientID': tensor([1.2]) Normalized
        true_value = patient_to_reg_value.get(pid)
        true_value = float(true_value) if true_value is not None else None # ie. 'PatientID': tensor([0.999) Normalized
        err = abs(pred_value - true_value) if (true_value is not None) else None

        rows.append({
            patient_label: pid,
            ground_truth_label: true_value,
            "pred": pred_value,
            "error": err,
        })
    df = pd.DataFrame(rows)
    # Sort descending by error, to see biggest errors first
    #df.sort_values("error", ascending=False, inplace=True)
    return df

