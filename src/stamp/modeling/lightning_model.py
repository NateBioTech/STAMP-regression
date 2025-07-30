"""Lightning wrapper around the model"""

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import lightning
import numpy as np
import torch
from jaxtyping import Bool, Float
from packaging.version import Version
from torch import Tensor, nn, optim
from torchmetrics.regression import MeanSquaredError, R2Score


import stamp
from stamp.modeling.data import (
    Bags,
    BagSizes,
    #Category,
    CoordinatesBatch,
    EncodedTargets,
    PandasLabel,
    PatientId,
)
from stamp.modeling.vision_transformer import VisionTransformer

Loss: TypeAlias = Float[Tensor, ""]


class LitVisionTransformer(lightning.LightningModule):
    def __init__(
        self,
        *,
        #categories: Sequence[Category],
        #category_weights: Float[Tensor, "category_weight"],  # noqa: F821
        dim_input: int,
        dim_model: int,
        dim_feedforward: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        # Experimental features
        # TODO remove default values for stamp 3; they're only here for backwards compatibility
        use_alibi: bool = False,
        # Metadata used by other parts of stamp, but not by the model itself
        ground_truth_label: PandasLabel,
        train_patients: Iterable[PatientId],
        valid_patients: Iterable[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        # Other metadata
        **metadata,
    ) -> None:
        """
        Args:
            metadata:
                Any additional information to be saved in the models,
                but not directly influencing the model.
        """
        super().__init__()


        self.vision_transformer = VisionTransformer(
            dim_output=1,
            dim_input=dim_input,
            dim_model=dim_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_alibi=use_alibi,
        )

        # Regression loss and metrics
        self.loss_fn = nn.MSELoss()
        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
        self.valid_r2 = R2Score()


        #self.class_weights = category_weights

        # Check if version is compatible.
        # This should only happen when the model is loaded,
        # otherwise the default value will make these checks pass.
        if stamp_version < Version("2.0.0.dev8"):
            # Update this as we change our model in incompatible ways!
            raise ValueError(
                f"model has been built with stamp version {stamp_version} "
                f"which is incompatible with the current version."
            )
        elif stamp_version > Version(stamp.__version__):
            # Let's be strict with models "from the future",
            # better fail deadly than have broken results.
            raise ValueError(
                "model has been built with a stamp version newer than the installed one "
                f"({stamp_version} > {stamp.__version__}). "
                "Please upgrade stamp to a compatible version."
            )

        #self.valid_auroc = MulticlassAUROC(len(categories))

        # Used during deployment
        self.ground_truth_label = ground_truth_label
        #self.categories = np.array(categories)
        self.train_patients = train_patients
        self.valid_patients = valid_patients

        _ = metadata  # unused, but saved in model

        self.save_hyperparameters()

    #Changed function forward
    def forward(
        self,
        bags: Bags,
    ) -> Float[Tensor, "batch logit"]:
        return self.vision_transformer(bags)


    def _step(
        self,
        *,
        step_name: str,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        _ = batch_idx  # unused

        bags, coords, bag_sizes, targets = batch

        logits = self.vision_transformer(
            bags, coords=coords, mask=_mask_from_bags(bags=bags, bag_sizes=bag_sizes)
        )
        
        logits = logits.squeeze(-1)  # Ensure correct shape for regression
        #print(logits)

        loss = self.loss_fn(logits, targets)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        #if step_name == "training":
        #    self.train_mse.update(logits, targets)
        #    self.log("train_mse", self.train_mse, on_epoch=True, prog_bar=True)

        if step_name == "validation":
            # TODO this is a bit ugly, we'd like to have `_step` without special cases
            #self.valid_mse.update(logits, targets)
            #self.valid_r2.update(logits, targets)
            #self.log("val_mse", self.valid_mse, on_epoch=True, prog_bar=True)
            #self.log("val_r2", self.valid_r2, on_epoch=True, prog_bar=True)

            self.valid_mse.update(logits, targets)
            self.valid_r2.update(logits, targets)
            self.log_dict({                   
                "val_mse": self.valid_mse,                    
                "val_r2":  self.valid_r2,},                
                on_epoch=True,                
                prog_bar=True,                
                sync_dist=True,)
        return loss


    def training_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(
            step_name="training",
            batch=batch,
            batch_idx=batch_idx,
        )

    def validation_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(
            step_name="validation",
            batch=batch,
            batch_idx=batch_idx,
        )
    
      

    def test_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(
            step_name="test",
            batch=batch,
            batch_idx=batch_idx,
        )

    def predict_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int = -1,
    ) -> Float[Tensor, "batch logit"]:
        bags, coords, bag_sizes, _ = batch
        return self.vision_transformer(
            bags, coords=coords, mask=_mask_from_bags(bags=bags, bag_sizes=bag_sizes)
        )
    

    #def configure_optimizers(self) -> optim.Optimizer:
    #    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer
    # ------------------------------------------------------------------
    # Testing 
    def configure_optimizers(self):
        """AdamW + CosineAnnealingLR (as Marugoto)."""
        opt = torch.optim.AdamW(
            self.parameters(), lr=3e-4, weight_decay=1e-2
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=30, eta_min=1e-5
        )
        return [opt], [sch]
    # ---------------------------------------------------

def _mask_from_bags(
    *,
    bags: Bags,
    bag_sizes: BagSizes,
) -> Bool[Tensor, "batch tile"]:
    max_possible_bag_size = bags.size(1)
    mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(0).repeat(
        len(bags), 1
    ) >= bag_sizes.unsqueeze(1)

    return mask
