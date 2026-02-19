from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseTrainer, Batch, TrainConfig
from .utils import OptimConfig, export_artifacts
from .losses import DSMLoss


class ContinuousTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optim_cfg: OptimConfig,
        train_cfg: TrainConfig,
        *,
        dsm_sigma: float = 1.0,
        n_noise: int = 1,
        temperature: float = 1.0,
        export: bool = True,
        export_model_path: Optional[str] = None,
        export_state_dict_path: Optional[str] = None,
        df=None,
        feature_names=None,
    ) -> None:
        super().__init__(model=model, optim_cfg=optim_cfg, train_cfg=train_cfg)

        setattr(self.train_cfg, "eval_requires_grad", True)
        self.dsm_loss = DSMLoss(sigma=float(dsm_sigma), n_noise=int(n_noise), temperature=float(temperature))

        self.export_enabled = bool(export)
        self.export_model_path = export_model_path
        self.export_state_dict_path = export_state_dict_path
        self.export_df = df
        self.export_feature_names = feature_names
            
    def training_step(self, batch: Batch) -> torch.Tensor:
        xb, yb = batch
        loss = self.dsm_loss(self.model, xb, yb)
        return loss

    def validation_step(self, batch: Batch) -> torch.Tensor:
        return self.training_step(batch)
    
    def export_structure(self, result=None):
        if self.export_enabled:
            export_artifacts(
                model=self.model,
                export_model_path=self.export_model_path,
                export_state_dict_path=self.export_state_dict_path,
                df=self.export_df,
                feature_names=self.export_feature_names,
                result=result,
            )