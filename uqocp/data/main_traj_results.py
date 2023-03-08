import pickle
import os
from ase.io import Trajectory
from tqdm import tqdm
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc
from finetuna.utils import compute_with_calc
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoints', nargs='+', help='<Required> List of paths to checkpoints for use, defaults to a preset list for an ensemble', required=False)
    parser.add_argument("-p","--picklefile", help='<Optional> Path to the pickle file for preloading tags, defaults to OC_20_val_data.pkl', required=False)
    parser.add_argument("-d", "--distributions", nargs='+', help='<Optional> Distributions in the pickle file to be sampled. Defaults to sampling all distributions.', required=False)
    args = parser.parse_args()

    # choose checkpoints, defaults to five plus DFT
    checkpoints = args.checkpoints
    if checkpoints is None:
        checkpoints = [
            "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc/gemnet_oc_base_s2ef_2M.pt",
            "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc/gemnet_oc_base_s2ef_all.pt",
            "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc/gemnet_oc_base_s2ef_all_md.pt",
            "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc/gemnet_oc_large_s2ef_all_md.pt",
            "/home/jovyan/shared-scratch/joe/personal_checkpoints/ocp_public_3_3_2023/gemnet_oc_22/gnoc_oc22_oc20_all_s2ef.pt",
            None,
        ]
    print("checkpoints:")
    for c in checkpoints:
        print("\t" + str(c))

    # preload tags
    pkl_path = args.picklefile
    if pkl_path is None:
        pkl_path=__file__[:__file__.rindex("/")] + "/OC_20_val_data.pkl"
    with open(os.path.join(pkl_path), "rb") as f:
        df = pickle.load(f)
    print("pickle path: " + str(pkl_path))

    # choose sample distributions, defaults to all
    distributions = args.distributions
    if distributions is None:
        distributions = list(set(df.distribution.to_list()))
    print("distributions: " + str(distributions))


    # Now execute main script
    calcs_dict = {}
    for checkpoint_path in checkpoints:
        if checkpoint_path is not None:
            save_path = checkpoint_path.split("/")[-1].split(".")[0]
            os.makedirs(save_path, exist_ok=True)
            try:
                calcs_dict[save_path] = FinetunerCalc(checkpoint_path, mlp_params={
                    "cpu": False,
                    "optim": {
                        "batch_size":5
                    }
                })
            except:
                print("ERROR: hit exception initializing: " + str(checkpoint_path))
                quit()
        else:
            save_path = "dft"
            os.mkdir(save_path)
            calcs_dict[save_path] = None

    for d in ["id"]:
        trajids = df[df.distribution == d].random_id.tolist()
        for tid in tqdm(trajids, d):
            traj = Trajectory("/home/jovyan/shared-datasets/OC20/trajs/val_02_01/"+tid+".traj")
            for save_path, calc in calcs_dict.items():
                if calc is not None:
                    ml_forces = np.array([x.get_forces() for x in compute_with_calc(traj, calc)])
                else:
                    ml_forces = np.array([x.get_forces() for x in traj])
                np.save(save_path+"/"+tid, ml_forces)


    print("done")