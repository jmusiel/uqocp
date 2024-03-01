from typing import Dict

from ase import Atoms
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import os
import torch
from ase.calculators.calculator import Calculator
from ocpmodels.datasets import data_list_collater
import faiss
import pickle
from uqocp.distance.conformal_prediction import FlexibleNLL


class OCPCalculatorLatent(OCPCalculator):

    def __init__(
        self,
        checkpoint_path: str,
        latent_model: str="latent_gemnet_oc",
        latent_trainer: str="latent",
        return_embedding: bool=True, 
        cpu: bool=True,
        enforce_max_neighbors_strictly: bool=True,

    ):
        self.return_embedding = return_embedding

        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        config = checkpoint["config"]
        config["model"] = latent_model
        config["trainer"] = latent_trainer
        config["dataset"]["train"] = config["dataset"]
        config["model_attributes"]["enforce_max_neighbors_strictly"] = enforce_max_neighbors_strictly

        super().__init__(
            config_yml=config,
            checkpoint_path=checkpoint_path,
            cpu=cpu
        )
    
    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = self.a2g.convert(atoms)
        batch = data_list_collater([data_object], otf_graph=True)

        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )
        if self.trainer.name == "s2ef":
            self.results["energy"] = predictions["energy"].item()
            self.results["forces"] = predictions["forces"].cpu().numpy()
            self.results["latents"] = predictions["latents"].cpu().numpy()

        elif self.trainer.name == "is2re":
            self.results["energy"] = predictions["energy"].item()

class DistanceIndex:
    def __init__(self, fitted_index_dir, k=1, per_atom_approach="mean"):
        self.load_index(fitted_index_dir)
        self.load_std_scaler(fitted_index_dir)
        self.load_cp_model(fitted_index_dir)
        self.num_nearest_neighbors = k
        self.per_atom_approach = per_atom_approach
    
    def load_index(self, load_dir):
        self.index = faiss.read_index(os.path.join(load_dir, "faiss.index"))
        print(f"loaded index from {load_dir}", flush=True)

    def load_std_scaler(self, load_dir):
        with open(os.path.join(load_dir, "std_scaler.pkl"), "rb") as f:
            self.std_scaler = pickle.load(f)
        print(f"loaded std scaler from {load_dir}", flush=True)

    def load_cp_model(self, load_dir):
        with open(os.path.join(load_dir, "cp_model.pkl"), "rb") as f:
            self.model_cp = pickle.load(f)
        print(f"loaded cp model from {load_dir}", flush=True)

    def calc_dist(self, latents):
        latents_std_scaled = self.std_scaler.transform(latents)
        distances, neighbors = self.index.search(latents_std_scaled, self.num_nearest_neighbors)
        if self.per_atom_approach == "mean":
            per_sys_distances = distances.mean()
        elif self.per_atom_approach == "max":
            per_sys_distances = distances.max()
        elif self.per_atom_approach == "sum":
            per_sys_distances = distances.sum()
        else:
            raise ValueError(f"per_atom_approach {self.per_atom_approach} not recognized")
        return per_sys_distances

    def predict(self, latents):
        distances = self.calc_dist(latents)
        uncertainty, qhat = self.model_cp.predict(distances)
        return uncertainty

