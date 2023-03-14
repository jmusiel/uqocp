import pickle
import numpy as np
import os
from tqdm import tqdm
import json
from finetuna.utils import force_l2_norm_err
import argparse
import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoints', nargs='+', help='<Required> List of paths to checkpoints for use, defaults to a preset list for an ensemble', required=False)
    parser.add_argument("-p","--picklefile", help='<Optional> Path to the pickle file for preloading tags, defaults to OC_20_val_data.pkl', required=False)
    parser.add_argument("-d", "--distributions", nargs='+', help='<Optional> Distributions in the pickle file to be sampled. Defaults to sampling all distributions.', required=False)
    parser.add_argument("-j", "--json", help="<Optional> choose json save file name (without extension), always prepended with '*_' where * is the distribution name, defaults to '*_errors.json'", required=False)
    parser.add_argument("-l", "--limit", type=int, help="<Optional> choose limited number of files to scan, defaults to all files in distribution")
    args = parser.parse_args()

    # choose checkpoints, defaults to five plus DFT
    checkpoints = args.checkpoints
    if checkpoints is None:
        checkpoints = [
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/dft",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/gemnet_oc_base_s2ef_2M",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/gemnet_oc_base_s2ef_all",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/gemnet_oc_base_s2ef_all_md",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/gemnet_oc_large_s2ef_all_md",
            "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/gnoc_oc22_oc20_all_s2ef",
        ]
    print("checkpoints:")
    for c in checkpoints:
        print("\t" + str(c))

    # preload tags
    pkl_path = args.picklefile
    if pkl_path is None:
        pkl_path=__file__[:__file__.rindex("/analysis/")] + "/data/OC_20_val_data.pkl"
    # with open(os.path.join(pkl_path), "rb") as f:
    #     df = pickle.load(f)
    print("pickle path: " + str(pkl_path))

    # choose sample distributions, defaults to all
    distributions = args.distributions
    if distributions is None:
        distributions = ["ft2_zeolites"]
    print("distributions: " + str(distributions))

    # choose json save file name, defaults to '*_errors.json' where * is the distribution name
    json_save_name = args.json
    if json_save_name is None:
        json_save_name = "errors"

    # choose limited number of files to scan, defaults to all files in distribution
    limit = args.limit

    i = 0
    # for d in distributions:
    for d in distributions:
        per_atom_dict = {model.split("/")[-1]:[] for model in checkpoints[1:]}
        # per_atom_dict["en_mean"] = []
        per_atom_dict["en_error"] = []
        per_atom_dict["en_stdev"] = []
        # trajids = df[df.distribution == d].random_id.tolist()
        trajpaths = [name for name in glob.glob("/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/MOFs_zeolites/Sudheesh_mof_zeolite/**/abinitio/oal_relaxation.traj", recursive=True)]
        if limit is not None:
            trajpaths = trajpaths[0:limit]      
        for trajpath in tqdm(trajpaths, d):
            tid = trajpath.replace("/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/MOFs_zeolites/Sudheesh_mof_zeolite/","").replace("/abinitio/oal_relaxation.traj","").replace("/","_")
            tpaths = [os.path.join(model, tid+".npy") for model in checkpoints]

            all_exist = True
            for tpath in tpaths:
                if not os.path.exists(tpath):
                    all_exist = False
                    break

            if all_exist:
                # for every atom in every frame in every system, record the following:
                # mean prediction of ensemble
                # stdev of ensemble prediction
                # error of mean prediction of ensemble
                # error of each individual model
                # distribution
                i = i+1 #
                forces_array = np.array([np.load(tpath).reshape(-1, 3) for tpath in tpaths])
                forces_array = forces_array[:, np.all(forces_array != 0, axis=(0,2))]
                dft_forces = forces_array[0]
                forces_array = forces_array[1:]
                mean_force = np.mean(forces_array, axis=0)
                mean_error = np.linalg.norm(dft_forces - mean_force, axis=1)
                stdev = np.zeros(mean_force.shape[0])
                for arr, model in zip(forces_array, checkpoints[1:]):
                    diff_array = arr - mean_force
                    l2_norm_vector = np.linalg.norm(diff_array, axis=1)
                    for n in l2_norm_vector:
                        per_atom_dict[model.split("/")[-1]].append(n)
                    sq_l2_norm_vector = np.square(l2_norm_vector)
                    stdev = stdev + sq_l2_norm_vector
                stdev = np.sqrt(stdev/forces_array.shape[0])

                for m, e, s in zip(mean_force, mean_error, stdev):
                    # per_atom_dict["en_mean"].append(m.tolist())
                    per_atom_dict["en_error"].append(e)
                    per_atom_dict["en_stdev"].append(s)

                # print the per model mean force error
                # for arr, model in zip(forces_array, model_paths[1:]):
                #     mae = np.mean(np.linalg.norm(dft_forces - arr, axis=1))
                #     print(model.split("/")[-1] + " MAE: " + str(mae))
                # print("\n")

        with open(d + '_' + json_save_name + '_ft2.json', "w") as json_file:
            json.dump(per_atom_dict, json_file)