import argparse
import pprint
pp = pprint.PrettyPrinter(indent=4)

import matplotlib.pyplot as plt
from ase import Atom, Atoms
import numpy as np
import os
import json
from ase.io import Trajectory, read

from uqocp.utils.ocp_calculator_latent import OCPCalculatorLatent, DistanceIndex
from uqocp.uncertainty_quantification.conformal_prediction import FlexibleNLL
from uqocp.utils.uncertainty_evaluation import recalibrate

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str, 
        default="/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc/gemnet_oc_large_s2ef_all_md.pt",
        help="absolute path to checkpoint, must be checkpoint used to make the distance index",
    )
    parser.add_argument(
        "--distance_index_path",
        type=str, 
        default="/home/jovyan/shared-scratch/joe/latent_distance_indexes/gnoc_per_atom_mean_latent_calibrated_on_eq2_err",
        help="absolute path to directory containing faiss.index file, along with cp_model.pkl file, and std_scaler.pkl file. These should have been created by uqocp/distance/conformal_prediction.py from the corresponding checkpoint.",
    )
    parser.add_argument(
        "--calibration_json",
        type=str, 
        default="/home/jovyan/shared-scratch/joe/latent_distance_indexes/gnoc_per_atom_mean_latent_calibrated_on_eq2_err/gnoc_per_atom_mean_calibration_results.json",
        help="absolute path to calibration_results.json, generated by fitting RMSV to RMSE on OCP val id IS2RE data using the corresponding index and checkpoint",
    )
    parser.add_argument(
        "--per_atom_approach",
        type=str, 
        default="mean",
        help="string should be 'mean', 'max', or 'sum' matching the approach used to generate the calibration file",
    )
    parser.add_argument(
        "--trajs_dir",
        type=str,
        default="[REPO]/examples/pt_o_example/single_frame_trajs",
        help="path to directory containing trajs with frames to predict latent embedding",
    )
    parser.add_argument(
        "--vasp_results_dir",
        type=str, 
        default="[REPO]/examples/pt_o_example/vasp_out",
        help="path to directory containing vasp results for each of the corresponding frames",
    )
    parser.add_argument(
        "--vasp_reference_json",
        type=str, 
        default="[REPO]/examples/pt_o_example/vasp_reference.json",
        help="path to json file containing vasp reference energies for the slab and adsorbate used",
    )
    return parser.parse_args()


def main(config):
    for key in config.keys():
        if type(config[key]) == str and "[REPO]" in config[key]:
            config[key] = config[key].replace("[REPO]", os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
    pp.pprint(config)

    # read in the adslabs with vasp results, and the matching adslabs to predict energy and uncertainty with OCP
    ml_adslabs_list = []
    vasp_adslabs_list = []
    for traj_path in os.listdir(config["trajs_dir"]):
        ml_atoms = Trajectory(os.path.join(config["trajs_dir"], traj_path))[0]
        ml_adslabs_list.append(ml_atoms)
        vasp_atoms = read(os.path.join(config["vasp_results_dir"], traj_path.replace(".traj",""), "vasprun.xml"))
        vasp_adslabs_list.append(vasp_atoms) 

    # load reference energies for the slab and adsorbate, these will be added to OCP predicted energy to make it match the VASP calculated energy
    with open(config["vasp_reference_json"]) as f:
        vasp_reference_dict = json.load(f)
    
    # load the latent ocp calculator using the provided checkpoint
    calc = OCPCalculatorLatent(checkpoint_path=config["checkpoint_path"], cpu=False)

    # for each given frame, predict the energy and latent embedding with the OCP calculator
    # then use the energy to calculate the error,
    print("#### Calculating energies and latents ####", flush=True)
    latents = []
    ml_energy_list = []
    vasp_energy_list = []
    num_o_atoms_list = []
    for i, (ml_atoms, vasp_atoms) in enumerate(zip(ml_adslabs_list, vasp_adslabs_list)):
        num_o_atoms = sum(atom.symbol == 'O' for atom in ml_atoms)
        num_o_atoms_list.append(num_o_atoms)
        ml_atoms.set_calculator(calc)
        ml_energy = ml_atoms.get_potential_energy() + vasp_reference_dict[str(num_o_atoms)]
        ml_energy_list.append(ml_energy)
        latents.append(ml_atoms.calc.results["latents"])
        vasp_energy_list.append(vasp_atoms.get_potential_energy())

    # load distance index
    # and use the latent embedding to calculate the distance, and then the calibrated uncertainty
    print("#### Loading distance index ####", flush=True)
    distance_index = DistanceIndex(config["distance_index_path"], per_atom_approach=config["per_atom_approach"])
    distances = []
    print("#### Calculating distances ####", flush=True)
    for i, l in enumerate(latents):
        distances.append(distance_index.predict(l))

    # load the calibration json, generated by fitting RMSV to RMSE on OCP val id IS2RE data
    with open(config["calibration_json"]) as f:
        calibration_dict = json.load(f)
    calibration_coefficients = [calibration_dict['calibration_results']["slope"], calibration_dict['calibration_results']["intercept"]]

    # and use the distances to calculate the calibrated uncertainty
    distances = np.array(distances)
    uncertainties = recalibrate(distances, calibration_coefficients)

    # sort results
    sorted_indices = np.argsort(num_o_atoms_list)
    num_o_atoms_list = np.array(num_o_atoms_list)[sorted_indices]
    uncertainties = np.array(uncertainties)[sorted_indices]
    ml_energy_list = np.array(ml_energy_list)[sorted_indices]
    vasp_energy_list = np.array(vasp_energy_list)[sorted_indices]

    # print results
    for n, unc, ocp_energy, vasp_energy in zip(num_o_atoms_list, uncertainties, ml_energy_list, vasp_energy_list):
        print(
            f"{n}: \n" + \
            f"\tuncertainty: {unc} (eV)\n" + \
            f"\terror: {abs(vasp_energy-ocp_energy)} (eV)\n" + \
            f"\tOCP energy: {ocp_energy} (eV)\n" + \
            f"\tVASP energy: {vasp_energy} (eV)\n"
        )

if __name__ == "__main__":
    args = get_args()
    config = vars(args)
    main(config)
    print("done")
