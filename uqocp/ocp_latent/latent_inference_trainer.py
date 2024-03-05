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
# from .latent_ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer
from ocpmodels.trainers.forces_trainer import ForcesTrainer

# from experimental.jmusiel.accumulate_s2ef.save_results_util import accumulate_results
# from experimental.jmusiel.skip_trainer.skip_trainer_mixin import SkipTrainerMixin

from torch_geometric.data.batch import Batch


@registry.register_trainer("latent")
class LatentTrainer(ForcesTrainer):
    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image=True,
        results_file=None,
        disable_tqdm=False,
    ):
        partial_size = self.config["task"].get("partial_size", 2000)
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

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": [], "latents": []}
        identifier_progress_path = self.config["cmd"]["results_dir"].replace(self.config["cmd"]["timestamp_id"], self.config["cmd"]["identifier"])
        os.makedirs(identifier_progress_path, exist_ok=True)
        progress_file_path = os.path.join(identifier_progress_path, f"{self.name}_finished_ids.csv")
        if os.path.exists(progress_file_path):
            logging.info(f"ADVISORY: Found progress file at {progress_file_path}. Will skip any systems in this file.")
            with open(progress_file_path, "r") as progress_file:
                finished_ids = set(progress_file.read().split(","))
        else:
            finished_ids = set()
        newly_finished_ids = set()
        save_counter = len(finished_ids)%partial_size

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # before forward pass, do some things if we performing inference (per_image, not relaxation)
            if per_image:
                # save current state of predictions for parallelism before we decide whether to skip this batch
                if i % partial_size == 0 and not i == 0:
                    self.save_results(predictions, f"{results_file}_partial{save_counter}", keys=["energy", "forces", "chunk_idx", "latents"], finished_ids=finished_ids)
                    predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": [], "latents": []}
                    save_counter += 1
                # skip this batch if all systems have been predicted
                contains_new_sid = False
                check_systemids = [str(i) + "_" + str(j) for i, j in zip(batch_list[0].sid.tolist(), batch_list[0].fid.tolist())]
                for systemid in check_systemids:
                    if systemid not in finished_ids:
                        contains_new_sid = True
                        finished_ids.add(systemid)
                        newly_finished_ids.add(systemid)
                if not contains_new_sid:
                    logging.info(f"Skipping the following systems ids because all systems have been predicted: {check_systemids}")
                    continue
            # otherwise we continue with the forward pass
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list, inference=True)

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
                    latents = out["latents"].cpu().detach().to(torch.float32)
                else:
                    predictions["energy"].extend(
                        out["energy"].cpu().detach().to(torch.float16).numpy()
                    )
                    forces = out["forces"].cpu().detach().to(torch.float16)
                    latents = out["latents"].cpu().detach().to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [
                    force.numpy() for force in per_image_forces
                ]
                per_image_latents = torch.split(latents, batch_natoms.tolist())
                per_image_latents = [latent.numpy() for latent in per_image_latents]
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
                    _per_image_free_latents = [
                        latent[(fixed == 0).tolist()]
                        for latent, fixed in zip(
                            per_image_latents, _per_image_fixed
                        )
                    ]
                    _chunk_idx = np.array(
                        [
                            free_force.shape[0]
                            for free_force in _per_image_free_forces
                        ]
                    )
                    per_image_forces = _per_image_free_forces
                    per_image_latents = _per_image_free_latents
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["forces"].extend(per_image_forces)
                predictions["latents"].extend(per_image_latents)
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                predictions["latents"] = out["latents"].detach()
                if self.ema:
                    self.ema.restore()
                return predictions

        predictions["forces"] = np.array(predictions["forces"])
        predictions["chunk_idx"] = np.array(predictions["chunk_idx"])
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(predictions["id"])
        predictions["latents"] = np.array(predictions["latents"])
        self.save_results(
            predictions, f"{results_file}_partial{save_counter}", keys=["energy", "forces", "chunk_idx", "latents"], finished_ids=finished_ids
        )

        if self.ema:
            self.ema.restore()

        return predictions

    def _forward(self, batch_list, inference=False):
        # forward pass.
        if inference:
            oom = False
            try:
                if self.config["model_attributes"].get("regress_forces", True):
                    out_energy, out_forces, latent_rep = self.model(batch_list)
                else:
                    out_energy, latent_rep = self.model(batch_list)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch one at a time')
                    print(f"total number of atoms in batch: {batch_list[0].natoms.sum().item()}")
                    torch.cuda.empty_cache()
                    oom = True
                else:
                    raise e
            if oom:
                out_energy = torch.tensor([], device=self.device)
                out_forces = torch.tensor([], device=self.device)
                latent_rep = torch.tensor([], device=self.device)
                data_list = batch_list[0].to_data_list()
                for data in data_list:
                    subbatch = Batch.from_data_list([data])
                    if self.config["model_attributes"].get("regress_forces", True):
                        sub_energy, sub_forces, sub_latent = self.model([subbatch])
                        out_forces = torch.cat((out_forces, sub_forces))
                    else:
                        sub_energy, sub_latent = self.model([subbatch])
                    out_energy = torch.cat((out_energy, sub_energy))
                    latent_rep = torch.cat((latent_rep, sub_latent))
                
        else:
            if self.config["model_attributes"].get("regress_forces", True):
                out_energy, out_forces, latent_rep = self.model(batch_list)
            else:
                out_energy, latent_rep = self.model(batch_list)


        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {
            "energy": out_energy,
        }

        if self.config["model_attributes"].get("regress_forces", True):
            out["forces"] = out_forces

        out["latents"] = latent_rep

        return out

    def save_results(self, predictions, results_file, keys, finished_ids):
        logging.info(f"Saving results: starting save process from {distutils.get_rank()}")
        if results_file is None:
            logging.info(f"Saving results: results file is None, returning from {distutils.get_rank()}")
            return

        results_file_path = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{self.name}_{results_file}_{distutils.get_rank()}.npz",
        )
        np.savez_compressed(
            results_file_path,
            ids=predictions["id"],
            **{key: predictions[key] for key in keys},
        )
        identifier_progress_path = self.config["cmd"]["results_dir"].replace(self.config["cmd"]["timestamp_id"], self.config["cmd"]["identifier"])
        progress_file_path = os.path.join(identifier_progress_path, f"{self.name}_finished_ids_{distutils.get_rank()}.csv")
        if self.config["task"].get("write_progress_file", False):
            with open(progress_file_path, "w") as progress_file:
                progress_file.write(",".join(finished_ids))

        logging.info(f"Saving results: trying to sync from {distutils.get_rank()}")
        distutils.synchronize()
        if distutils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["cmd"]["results_dir"],
                f"{self.name}_{results_file}.npz",
            )
            gather_progress = set()

            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"{self.name}_{results_file}_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                for key in keys:
                    gather_results[key].extend(rank_results[key])
                os.remove(rank_path)

                if self.config["task"].get("write_progress_file", False):
                    rank_progress_path = os.path.join(identifier_progress_path, f"{self.name}_finished_ids_{i}.csv")
                    with open(rank_progress_path, "r") as progress_file:
                        gather_progress.update(progress_file.read().split(","))
                    os.remove(rank_progress_path)

            if len(gather_results["ids"]) == 0:
                return

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            gather_results["ids"] = np.array(gather_results["ids"])[idx]
            for k in keys:
                if k == "forces":
                    gather_results[k] = np.concatenate(
                        np.array(gather_results[k])[idx]
                    )
                elif k == "latents":
                    gather_results[k] = np.concatenate(
                        np.array(gather_results[k])[idx]
                    )
                elif k == "chunk_idx":
                    gather_results[k] = np.cumsum(
                        np.array(gather_results[k])[idx]
                    )[:-1]
                else:
                    gather_results[k] = np.array(gather_results[k])[idx]

            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

            if self.config["task"].get("write_progress_file", False):
                full_progress_path = os.path.join(identifier_progress_path, f"{self.name}_finished_ids.csv")
                temp_full_progress_path = os.path.join(identifier_progress_path, f"{self.name}_finished_ids_temp.csv")
                with open(temp_full_progress_path, "w") as progress_file:
                    progress_file.write(",".join(gather_progress))
                if os.path.exists(full_progress_path):
                    os.remove(full_progress_path)
                os.rename(temp_full_progress_path, full_progress_path)

            # save accumulated results
            if self.config["task"].get("accumulate", True):
                accumulated_path = os.path.join(identifier_progress_path, f"{self.name}_{results_file.split('_partial')[0]}.npz")
                logging.info(f"Writing accumulated results to {accumulated_path}")
                accumulate_results(
                    accumulate_npz_path=accumulated_path,
                    results_to_append=gather_results,
                    warning_id=f"{self.name}_{results_file}",
                )
            else:
                with open(os.path.join(identifier_progress_path, f"{self.name}_NO_ACCUMULATION"), "w") as f:
                    f.write("not accumulating, please collect results from separated files")

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