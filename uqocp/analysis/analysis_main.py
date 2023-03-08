import pickle
import numpy as np
import os
from tqdm import tqdm
import json
from finetuna.utils import force_l2_norm_err


if __name__ == "__main__":
    # preload tags
    pkl_path="/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/OC_20_val_data.pkl"
    with open(os.path.join(pkl_path), "rb") as f:
        df = pickle.load(f)

    distributions = list(set(df.distribution.to_list()))

    model_paths = [
        "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/dft",
        "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/gemnet_oc_base_s2ef_2M",
        "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/gemnet_oc_base_s2ef_all",
        "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/gemnet_oc_large_s2ef_all_md",
        # "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/gnoc_finetune_all_s2ef",
        # "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/gnoc_oc22_all_s2ef",
        "/home/jovyan/shared-scratch/joe/jobs/uncertainty/data/oc20_val/gnoc_oc22_oc20_all_s2ef",
    ]

    i = 0
    # for d in distributions:
    for d in ["id"]:
        per_atom_dict = {model.split("/")[-1]:[] for model in model_paths[1:]}
        # per_atom_dict["en_mean"] = []
        per_atom_dict["en_error"] = []
        per_atom_dict["en_stdev"] = []
        trajids = df[df.distribution == d].random_id.tolist()
        for tid in tqdm(trajids[0:2000], d):
            tpaths = [os.path.join(model, tid+".npy") for model in model_paths]

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
                for arr, model in zip(forces_array, model_paths[1:]):
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

        with open(d + '_errors.json', "w") as json_file:
            json.dump(per_atom_dict, json_file)
