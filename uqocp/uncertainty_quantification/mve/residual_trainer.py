
import logging
import os
import pathlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer

from ocpmodels.trainers.forces_trainer import ForcesTrainer
from ocpmodels.modules.loss import AtomwiseL2Loss, DDPLoss, L2MAELoss
import torch.nn as nn


@registry.register_trainer("residual")
class ResidualTrainer(ForcesTrainer):
    def train(self, disable_eval_tqdm=False):
        for name, param in self.model.named_parameters():
            if "residual" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return super().train(disable_eval_tqdm)
    
    def load_loss(self):
        super().load_loss()
        loss = "residual"
        loss_name = self.config["optim"].get("loss_residual", "mae")
        if loss_name in ["l1", "mae"]:
            self.loss_fn[loss] = nn.L1Loss()
        elif loss_name == "mse":
            self.loss_fn[loss] = nn.MSELoss()
        elif loss_name == "l2mae":
            self.loss_fn[loss] = L2MAELoss()
        elif loss_name == "atomwisel2":
            self.loss_fn[loss] = AtomwiseL2Loss()
        else:
            raise NotImplementedError(
                f"Unknown loss function name: {loss_name}"
            )
        self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])
        self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])


    def _forward(self, batch_list):
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True):
            out_energy, out_forces, out_residual = self.model(batch_list)
        else:
            out_energy, out_residual = self.model(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        if out_residual.shape[-1] == 1:
            out_residual = out_residual.view(-1)

        out = {
            "energy": out_energy,
            "residual": out_residual,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        return out
    
    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        # loss.append(
        #     energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
        # )

        # residual loss
        residual_target = out["energy"] - energy_target
        residual_target = residual_target.abs()
        residual_mult = self.config["optim"].get("residual_coefficient", 1)
        loss.append(
            residual_mult * self.loss_fn["residual"](out["residual"], residual_target)
        )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "forces": torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
        }

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(
                    torch.sum(mask[s_idx : s_idx + natoms]).item()
                )
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

        target["residual"] = out["energy"] - target["energy"]
        target["residual"] = target["residual"].abs()

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics


    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image=True,
        results_file=None,
        disable_tqdm=False,
    ):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": [], "residual": []}

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )
                out["forces"] = self.normalizers["grad_target"].denorm(
                    out["forces"]
                )
                # out["residual"] = self.normalizers["target"].denorm(
                #     out["residual"]
                # )
            if per_image:
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                # total energy target requires predictions to be saved in float32
                # default is float16
                if (
                    self.config["task"].get("prediction_dtype", "float16")
                    == "float32"
                    or self.config["task"]["dataset"] == "oc22_lmdb"
                ):
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float32).numpy()
                    )
                    predictions["residual"].extend(
                        out["residual"].cpu().detach().to(torch.float32).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float32)
                else:
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float16).numpy()
                    )
                    predictions["residual"].extend(
                        out["residual"].cpu().detach().to(torch.float16).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [
                    force.numpy() for force in per_image_forces
                ]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_forces = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(
                            per_image_forces, _per_image_fixed
                        )
                    ]
                    _chunk_idx = np.array(
                        [
                            free_force.shape[0]
                            for free_force in _per_image_free_forces
                        ]
                    )
                    per_image_forces = _per_image_free_forces
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["forces"].extend(per_image_forces)
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["residual"] = out["residual"].detach()
                predictions["forces"] = out["forces"].detach()
                if self.ema:
                    self.ema.restore()
                return predictions

        predictions["forces"] = np.array(predictions["forces"])
        predictions["chunk_idx"] = np.array(predictions["chunk_idx"])
        predictions["energy"] = np.array(predictions["energy"])
        predictions["residual"] = np.array(predictions["residual"])
        predictions["id"] = np.array(predictions["id"])
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx", "residual"]
        )

        if self.ema:
            self.ema.restore()

        return predictions

