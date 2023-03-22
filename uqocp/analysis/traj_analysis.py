import pickle
import numpy as np
import os
from tqdm import tqdm
import json
from finetuna.utils import force_l2_norm_err
import argparse
from pqdm.processes import pqdm
from ase.io import Trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--trajdirs', nargs='+', help='<Required> List of paths to directories of trajectories for use, defaults to a preset list for an ensemble', required=False)
    parser.add_argument("-p","--picklefile", help='<Optional> Path to the pickle file for preloading tags, defaults to OC_20_val_data.pkl', required=False)
    parser.add_argument("-d", "--distributions", nargs='+', help='<Optional> Distributions in the pickle file to be sampled. Defaults to sampling all distributions.', required=False)
    parser.add_argument("-j", "--json", help="<Optional> choose json save file name (without extension), always prepended with '*_' where * is the distribution name, defaults to '*_errors.json'", required=False)
    parser.add_argument("-l", "--limit", type=int, help="<Optional> choose limited number of files to scan, defaults to all files in distribution")
    args = parser.parse_args()

    # choose checkpoints, defaults to five plus DFT
    trajdirs = args.trajdirs
    if trajdirs is None:
        checkpoints = [
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/traj2traj/ocp_val/dft",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/traj2traj/ocp_val/gemnet_oc_base_s2ef_2M",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/traj2traj/ocp_val/gemnet_oc_base_s2ef_all",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/traj2traj/ocp_val/gemnet_oc_base_s2ef_all_md",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/traj2traj/ocp_val/gemnet_oc_large_s2ef_all_md",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/traj2traj/ocp_val/gnoc_oc22_oc20_all_s2ef",
        ]
    print("checkpoints:")
    for c in checkpoints:
        print("\t" + str(c))

    # preload tags
    pkl_path = args.picklefile
    if pkl_path is None:
        pkl_path=__file__[:__file__.rindex("/analysis/")] + "/data/OC_20_val_data.pkl"
    with open(os.path.join(pkl_path), "rb") as f:
        df = pickle.load(f)
    print("pickle path: " + str(pkl_path))

    # choose sample distributions, defaults to all
    distributions = args.distributions
    if distributions is None:
        distributions = list(set(df.distribution.to_list()))
    print("distributions: " + str(distributions))

    # choose json save file name, defaults to '*_errors.json' where * is the distribution name
    json_save_name = args.json
    if json_save_name is None:
        json_save_name = "errors"

    # choose limited number of files to scan, defaults to all files in distribution
    limit = args.limit

    def get_system_errors(tid):
        temp_dict = {}
        # tpaths = [os.path.join(model, tid+".npy") for model in checkpoints]
        pathsdict = {}
        pathsdict["inf"] = [os.path.join(model, tid+".traj") for model in checkpoints]
        pathsdict["is2re"] = [os.path.join(model, tid+"_is2re.traj") for model in checkpoints[1:]]
        pathsdict["ens"] = [os.path.join(model, tid+"_ens.traj") for model in checkpoints]

        exist_dict = {
            "inf": True,
            "is2re": True,
            "ens": True,
        }
        all_exist = True
        for key, value in pathsdict.items():
            for tpath in value:
                if not os.path.exists(tpath):
                    all_exist = False
                    exist_dict[key] = False
                    break

        
        if exist_dict["inf"]:
            trajs = [[image for image in Trajectory(tpath)] for tpath in pathsdict["inf"]]
            energy_array = np.array([[traj[0].get_potential_energy() - image.get_potential_energy() for image in traj] for traj in trajs])
            dft_energy = energy_array[0]
            energy_array = energy_array[1:]
            mean_energy = np.mean(energy_array, axis=0)
            energy_error = dft_energy - mean_energy
            energy_std = np.std(energy_array, axis=0)
            temp_dict["inf_e_error"] = energy_error.tolist()
            temp_dict["inf_e_stdev"] = energy_std.tolist()
            temp_dict["dft_de"] = dft_energy[0] - dft_energy[-1]
            temp_dict["inf_de_mean"] = mean_energy[0] - mean_energy[-1]

            forces_array = np.array([[image.get_forces() for image in traj] for traj in trajs])
            forces_array = forces_array[:, :, np.all(forces_array != 0, axis=(0,1,3))]
            dft_forces = forces_array[0]
            forces_array = forces_array[1:]
            mean_force = np.mean(forces_array, axis=0)
            mean_error = np.linalg.norm(dft_forces - mean_force, axis=2)
            stdev = np.zeros_like(mean_error)
            for arr in forces_array:
                diff_array = arr - mean_force
                l2_norm_vector = np.linalg.norm(diff_array, axis=2)
                sq_l2_norm_vector = np.square(l2_norm_vector)
                stdev = stdev + sq_l2_norm_vector
            stdev = np.sqrt(stdev/forces_array.shape[0])
            temp_dict["inf_f_l2error"] = mean_error.tolist()
            temp_dict["inf_f_stdev"] = stdev.tolist()

        if exist_dict["is2re"]:
            trajs = [[image for image in Trajectory(tpath)] for tpath in pathsdict["is2re"]]
            deltae = [traj[0].get_potential_energy() - traj[-1].get_potential_energy() for traj in trajs]
            temp_dict["is2re_de_error"] = (dft_energy[0] - dft_energy[-1]) - np.mean(deltae)
            temp_dict["is2re_de_stdev"] = np.std(deltae)

        if exist_dict["ens"]:
            trajs = [[image for image in Trajectory(tpath)] for tpath in pathsdict["ens"]]
            energy_array = np.array([[traj[0].get_potential_energy() - image.get_potential_energy() for image in traj] for traj in trajs])
            # dft_energy = energy_array[0]
            energy_array = energy_array[1:]
            mean_energy = np.mean(energy_array, axis=0)
            # energy_error = dft_energy - mean_energy
            energy_std = np.std(energy_array, axis=0)
            temp_dict["ens_e_stdev"] = energy_std.tolist()
            temp_dict["ens_de_mean"] = mean_energy[0] - mean_energy[-1]

            forces_array = np.array([[image.get_forces() for image in traj] for traj in trajs])
            forces_array = forces_array[:, :, np.all(forces_array != 0, axis=(0,1,3))]
            # dft_forces = forces_array[0]
            forces_array = forces_array[1:]
            mean_force = np.mean(forces_array, axis=0)
            # mean_error = np.linalg.norm(dft_forces - mean_force, axis=2)
            stdev = np.zeros_like(np.mean(mean_force, axis=2))
            for arr in forces_array:
                diff_array = arr - mean_force
                l2_norm_vector = np.linalg.norm(diff_array, axis=2)
                sq_l2_norm_vector = np.square(l2_norm_vector)
                stdev = stdev + sq_l2_norm_vector
            stdev = np.sqrt(stdev/forces_array.shape[0])
            temp_dict["ens_f_stdev"] = stdev.tolist()

        return temp_dict

    i = 0
    # for d in distributions:
    for d in distributions:
        # system_dict = {model.split("/")[-1]:[] for model in checkpoints[1:]}
        # # system_dict["en_mean"] = []
        # system_dict["en_error"] = []
        # system_dict["en_stdev"] = []
        system_dict = {
            "inf_e_error": [],
            "inf_e_stdev": [],
            "dft_de": [],
            "inf_de_mean": [],
            "inf_f_l2error": [],
            "inf_f_stdev": [],
            "is2re_de_error": [],
            "is2re_de_stdev": [],
            "ens_e_stdev": [],
            "ens_de_mean": [],
            "ens_f_stdev": [],
        }
        trajids = df[df.distribution == d]
        trajids = [row["random_id"] for i, row in trajids.iterrows()]
        if limit is not None:
            trajids = trajids[:limit]

        result_list = pqdm(trajids, get_system_errors, n_jobs=8)
        # result_list = []
        # for tid in tqdm(trajids):
        #     result_list.append(get_system_errors(tid))
        for key, value in system_dict.items():
            for result in result_list:
                if result:
                    value.append(result[key])
                

        print("done")

        with open(d + '_' + json_save_name + '_traj2traj.json', "w") as json_file:
            json.dump(system_dict, json_file)
