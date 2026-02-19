from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseTrainer, Batch, TrainConfig
from .utils import OptimConfig, export_artifacts
from .losses import DiscreteCELoss


class DiscreteTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optim_cfg: OptimConfig,
        train_cfg: TrainConfig,
        extract_fn=None,
        *,
        export: bool = False,
        export_model_path: Optional[str] = None,
        export_state_dict_path: Optional[str] = None,
        df=None,
        feature_names=None,
    ) -> None:
        super().__init__(model=model, optim_cfg=optim_cfg, train_cfg=train_cfg, extract_fn=extract_fn)
        self.loss_fn = DiscreteCELoss()

        self.export_enabled = bool(export)
        self.export_model_path = export_model_path
        self.export_state_dict_path = export_state_dict_path
        self.df = df
        self.feature_names = feature_names

    def training_step(self, batch: Batch) -> torch.Tensor:
        x, y = batch
        bl_vec = self.model.logits(x)   
        return self.loss_fn(bl_vec, y)    

    def validation_step(self, batch: Batch) -> torch.Tensor:
        return self.training_step(batch)
    
    def export_structure(self, result=None):
        if self.export_enabled:
            export_artifacts(
                model=self.model,
                export_model_path=self.export_model_path,
                export_state_dict_path=self.export_state_dict_path,
                df=self.df,
                feature_names=self.feature_names,
                result=result,
            )
