

"""
exact copy of methods from:
    ocpmodels.common.relaxation.ml_relaxation
    ocpmodels.common.relaxation.optimizers.lbfgs_torch
all so I can replace the writing functionality of LBFGS.write to include uncertainty results in .traj files
and allow that information to pass through TorchCalc.get_energy_and_forces to LBFGS.write
"""

import logging
from collections import deque
from pathlib import Path

import torch
from torch_geometric.data import Batch

from ocpmodels.common.registry import registry
from ocpmodels.datasets.lmdb_dataset import data_list_collater

from ocpmodels.common.utils import radius_graph_pbc
from typing import Deque, Optional
import ase
from torch_scatter import scatter
from ocpmodels.common.relaxation.ase_utils import batch_to_atoms

from ocpmodels.common import distutils


def ml_relax(
    batch,
    model,
    steps,
    fmax,
    relax_opt,
    save_full_traj,
    device="cuda:0",
    transform=None,
    early_stop_batch=False,
):
    """
    Runs ML-based relaxations.
    Args:
        batch: object
        model: object
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
        save_full_traj: bool
            Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
    """
    batches = deque([batch[0]])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid
        calc = TorchCalc(model, transform)

        # Run ML-based relaxation
        traj_dir = relax_opt.get("traj_dir", None)
        optimizer = LBFGS(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            memory=relax_opt["memory"],
            damping=relax_opt.get("damping", 1.0),
            alpha=relax_opt.get("alpha", 70.0),
            device=device,
            save_full_traj=save_full_traj,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
            early_stop_batch=early_stop_batch,
        )
        try:
            relaxed_batch = optimizer.run(fmax=fmax, steps=steps)
            relaxed_batches.append(relaxed_batch)
        except RuntimeError as e:
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # move OOM recovery code outside of except clause to allow tensors to be freed.
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise e
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(data_list_collater(data_list[:mid]))
            batches.appendleft(data_list_collater(data_list[mid:]))

    relaxed_batch = Batch.from_data_list(relaxed_batches)
    return relaxed_batch

class LBFGS:
    def __init__(
        self,
        batch: Batch,
        model: "TorchCalc",
        maxstep=0.01,
        memory=100,
        damping=0.25,
        alpha=100.0,
        force_consistent=None,
        device="cuda:0",
        save_full_traj=True,
        traj_dir: Path = None,
        traj_names=None,
        early_stop_batch: bool = False,
    ):
        self.batch = batch
        self.model = model
        self.maxstep = maxstep
        self.memory = memory
        self.damping = damping
        self.alpha = alpha
        self.H0 = 1.0 / self.alpha
        self.force_consistent = force_consistent
        self.device = device
        self.save_full = save_full_traj
        self.traj_dir = traj_dir
        self.traj_names = traj_names
        self.early_stop_batch = early_stop_batch
        self.otf_graph = model.model._unwrapped_model.otf_graph
        assert not self.traj_dir or (
            traj_dir and len(traj_names)
        ), "Trajectory names should be specified to save trajectories"
        logging.info("Step   Fmax(eV/A)")

        if not self.otf_graph and "edge_index" not in batch:
            self.model.update_graph(self.batch)

    def get_energy_and_forces(self, apply_constraint=True):
        energy, forces, unc_dict = self.model.get_energy_and_forces(
            self.batch, apply_constraint
        )
        return energy, forces, unc_dict

    def set_positions(self, update, update_mask):
        if not self.early_stop_batch:
            update = torch.where(update_mask.unsqueeze(1), update, 0.0)
        self.batch.pos += update.to(dtype=torch.float32)

        if not self.otf_graph:
            self.model.update_graph(self.batch)

    def check_convergence(self, iteration, forces=None, energy=None):
        if forces is None or energy is None:
            energy, forces, unc_dict = self.get_energy_and_forces()
            forces = forces.to(dtype=torch.float64)

        max_forces_ = scatter(
            (forces**2).sum(axis=1).sqrt(), self.batch.batch, reduce="max"
        )
        logging.info(
            f"{iteration} "
            + " ".join(f"{x:0.3f}" for x in max_forces_.tolist())
        )

        # (batch_size) -> (nAtoms)
        max_forces = max_forces_[self.batch.batch]

        return max_forces.ge(self.fmax), energy, forces, unc_dict

    def run(self, fmax, steps):
        self.fmax = fmax
        self.steps = steps

        self.s = deque(maxlen=self.memory)
        self.y = deque(maxlen=self.memory)
        self.rho = deque(maxlen=self.memory)
        self.r0 = self.f0 = None

        self.trajectories = None
        if self.traj_dir:
            self.traj_dir.mkdir(exist_ok=True, parents=True)
            self.trajectories = [
                ase.io.Trajectory(self.traj_dir / f"{name}_{distutils.get_rank()}.traj_tmp", mode="w") # TODO: remove distutils from path
                for name in self.traj_names
            ]

        iteration = 0
        converged = False
        while iteration < steps and not converged:
            update_mask, energy, forces, unc_dict = self.check_convergence(iteration)
            converged = torch.all(torch.logical_not(update_mask))

            if self.trajectories is not None:
                if (
                    self.save_full
                    or converged
                    or iteration == steps - 1
                    or iteration == 0
                ):
                    if distutils.get_rank() == 0:
                        self.write(energy, forces, update_mask, unc_dict)

            if not converged and iteration < steps - 1:
                self.step(iteration, forces, update_mask)

            iteration += 1

        # GPU memory usage as per nvidia-smi seems to gradually build up as
        # batches are processed. This releases unoccupied cached memory.
        torch.cuda.empty_cache()

        if self.trajectories is not None:
            for traj in self.trajectories:
                traj.close()
            for name in self.traj_names:
                traj_fl = Path(self.traj_dir / f"{name}_{distutils.get_rank()}.traj_tmp", mode="w") # TODO: remove distutils from path
                if distutils.get_rank() == 0:
                    traj_fl.rename(traj_fl.with_suffix(".traj"))

        self.batch.y, self.batch.force, unc_dict = self.get_energy_and_forces(
            apply_constraint=False
        )
        return self.batch

    def step(
        self,
        iteration: int,
        forces: Optional[torch.Tensor],
        update_mask: torch.Tensor,
    ):
        def determine_step(dr):
            steplengths = torch.norm(dr, dim=1)
            longest_steps = scatter(
                steplengths, self.batch.batch, reduce="max"
            )
            longest_steps = longest_steps[self.batch.batch]
            maxstep = longest_steps.new_tensor(self.maxstep)
            scale = (longest_steps + 1e-7).reciprocal() * torch.min(
                longest_steps, maxstep
            )
            dr *= scale.unsqueeze(1)
            return dr * self.damping

        if forces is None:
            _, forces = self.get_energy_and_forces()

        r = self.batch.pos.clone().to(dtype=torch.float64)

        # Update s, y, rho
        if iteration > 0:
            s0 = (r - self.r0).flatten()
            self.s.append(s0)

            y0 = -(forces - self.f0).flatten()
            self.y.append(y0)

            self.rho.append(1.0 / torch.dot(y0, s0))

        loopmax = min(self.memory, iteration)
        alpha = forces.new_empty(loopmax)
        q = -forces.flatten()

        for i in range(loopmax - 1, -1, -1):
            alpha[i] = self.rho[i] * torch.dot(self.s[i], q)  # b
            q -= alpha[i] * self.y[i]

        z = self.H0 * q
        for i in range(loopmax):
            beta = self.rho[i] * torch.dot(self.y[i], z)
            z += self.s[i] * (alpha[i] - beta)

        # descent direction
        p = -z.reshape((-1, 3))
        dr = determine_step(p)
        if torch.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        self.set_positions(dr, update_mask)

        self.r0 = r
        self.f0 = forces

    def write(self, energy, forces, update_mask, unc_dict): # energy should be a tensor: "tensor([-1.0269], device='cuda:1')"
        self.batch.y, self.batch.force = energy, forces
        atoms_objects = batch_to_atoms(self.batch)
        update_mask_ = torch.split(update_mask, self.batch.natoms.tolist())
        for atm, traj, mask in zip(
            atoms_objects, self.trajectories, update_mask_
        ):
            for key, value in unc_dict.items():
                if not (key == "energy" or key == "forces" or key == "chunk_idx"):
                    if type(value) == torch.Tensor:
                        atm.info[key] = value.tolist()
                    else:
                        atm.info[key] = value
            if mask[0] or not self.save_full:
                traj.write(atm)


class TorchCalc:
    def __init__(self, model, transform=None):
        self.model = model
        self.transform = transform

    def get_energy_and_forces(self, atoms, apply_constraint=True):
        predictions = self.model.predict(
            atoms, per_image=False, disable_tqdm=True
        )
        energy = predictions["energy"]
        forces = predictions["forces"]
        if apply_constraint:
            fixed_idx = torch.where(atoms.fixed == 1)[0]
            forces[fixed_idx] = 0
        unc_dict = predictions
        return energy, forces, unc_dict

    def update_graph(self, atoms):
        edge_index, cell_offsets, num_neighbors = radius_graph_pbc(
            atoms, 6, 50
        )
        atoms.edge_index = edge_index
        atoms.cell_offsets = cell_offsets
        atoms.neighbors = num_neighbors
        if self.transform is not None:
            atoms = self.transform(atoms)
        return atoms