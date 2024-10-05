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
from .ensemble_lbfgs import ml_relax
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer
from ocpmodels.trainers.forces_trainer import ForcesTrainer
from ocpmodels.common.gp_utils import _gather

import datetime
import errno
import logging
import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import ocpmodels
from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict, save_checkpoint
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.exponential_moving_average import (
    ExponentialMovingAverage,
)
from ocpmodels.modules.loss import AtomwiseL2Loss, DDPLoss, L2MAELoss
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.compat import load_scales_compat
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.modules.scheduler import LRScheduler


def check_traj_files(batch, traj_dir):
    if traj_dir is None:
        return False
    traj_dir = Path(traj_dir)
    traj_files = [traj_dir / f"{id}.traj" for id in batch[0].sid.tolist()]
    traj_files0 = [traj_dir / f"{id}_0.traj" for id in batch[0].sid.tolist()]
    return all(fl.exists() for fl in traj_files) or all(fl.exists() for fl in traj_files0)


@registry.register_trainer("ensemble")
class EnsembleTrainer(ForcesTrainer):
    def get_sampler(self, dataset, batch_size, shuffle):
        if "load_balancing" in self.config["optim"]:
            balancing_mode = self.config["optim"]["load_balancing"]
            force_balancing = True
        else:
            balancing_mode = "atoms"
            force_balancing = False

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=1,
            rank=0,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )
        return sampler
    
    def load_model(self):
        self.checkpoint_paths = self.config["task"]["ensemble"]["checkpoint_paths"]
        current_checkpoint = torch.load(self.checkpoint_paths[distutils.get_rank()])
        self.config["model"] = current_checkpoint["config"]["model"]
        self.config["model_attributes"] = current_checkpoint["config"]["model_attributes"]

        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        torch.cuda.set_device(self.device)
        loader = self.train_loader or self.val_loader or self.test_loader
        self.model = registry.get_model_class(self.config["model"])(
            loader.dataset[0].x.shape[-1]
            if loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = self.checkpoint_paths[distutils.get_rank()]

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        strict = self.config["task"].get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

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

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": []}

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
                    forces = out["forces"].cpu().detach().to(torch.float32)
                else:
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float16).numpy()
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
                predictions["forces"] = out["forces"].detach()
                if self.ema:
                    self.ema.restore()

                gathered_energy = _gather(predictions["energy"])
                gathered_forces = _gather(predictions["forces"])
                gathered_forces = torch.stack(gathered_forces.split(3, dim=1))
                predictions["energy"] = torch.tensor([gathered_energy.mean(dim=0)], device=gathered_energy.device)
                predictions["forces"] = gathered_forces.mean(dim=0)

                energy_unc = torch.std(gathered_energy)
                diffs = torch.norm(predictions["forces"] - gathered_forces,dim=-1)
                diffs_n = torch.norm(predictions["forces"] - gathered_forces,dim=-1).shape[0]
                force_unc = torch.sqrt(torch.sum(torch.square(diffs), dim=0)/diffs_n)
                if "fid" in batch_list[0]:
                    systemids = [
                        str(i) + "_" + str(j)
                        for i, j in zip(
                            batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                        )
                    ]
                else:
                    systemids = [ str(i) for i in batch_list[0].sid.tolist()]
                
                predictions["all_energy"] = gathered_energy
                predictions["all_forces"] = gathered_forces
                predictions["force_unc"] = force_unc
                predictions["energy_unc"] = energy_unc
                predictions["id"] = systemids

                return predictions

        predictions["forces"] = np.array(predictions["forces"])
        predictions["chunk_idx"] = np.array(predictions["chunk_idx"])
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(predictions["id"])
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx"]
        )

        if self.ema:
            self.ema.restore()

        return predictions
    
    def run_relaxations(self, split="val"):
        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()

        registry.unregister("set_deterministic_scatter")