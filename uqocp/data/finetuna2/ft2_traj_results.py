import pickle
import os
from ase.io import Trajectory
from tqdm import tqdm
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc
from finetuna.utils import compute_with_calc
import numpy as np
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoints', nargs='+', help='<Required> List of paths to checkpoints for use, defaults to a preset list for an ensemble', required=False)
    # parser.add_argument("-p","--picklefile", help='<Optional> Path to the pickle file for preloading tags, defaults to OC_20_val_data.pkl', required=False)
    parser.add_argument("-d", "--distributions", nargs='+', help='<Optional> Distributions in the pickle file to be sampled. Defaults to sampling all distributions.', required=False)
    parser.add_argument("-s", "--skip", help='<Optional> Whether to skip files that exist already, if passed then false ', action="store_true")
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

    # # preload tags
    # pkl_path = args.picklefile
    # if pkl_path is None:
    #     pkl_path=__file__[:__file__.rindex("/")] + "/OC_20_val_data.pkl"
    # with open(os.path.join(pkl_path), "rb") as f:
    #     df = pickle.load(f)
    # print("pickle path: " + str(pkl_path))

    # choose sample distributions, defaults to all
    distributions = args.distributions
    if distributions is None:
        distributions = ["ft2_zeolites"]
    print("distributions: " + str(distributions))

    # choose whether to replace existing files, or skip inference on them
    skip = args.skip
    if skip is None:
        skip = False


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
            os.makedirs(save_path, exist_ok=True)
            calcs_dict[save_path] = None

    for d in distributions:
        # trajids = df[df.distribution == d].random_id.tolist()
        trajpaths = [name for name in glob.glob("/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/MOFs_zeolites/Sudheesh_mof_zeolite/**/abinitio/oal_relaxation.traj", recursive=True)]
        for trajpath in tqdm(trajpaths, d):
            traj = Trajectory(trajpath)
            tid = trajpath.replace("/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/MOFs_zeolites/Sudheesh_mof_zeolite/","").replace("/abinitio/oal_relaxation.traj","").replace("/","_")
            for save_path, calc in calcs_dict.items():
                if not skip or not os.path.isfile(save_path+"/"+tid+".npy"):
                    if calc is not None:
                        ml_forces = np.array([x.get_forces() for x in compute_with_calc(traj, calc)])
                    else:
                        ml_forces = np.array([x.get_forces() for x in traj])
                    np.save(save_path+"/"+tid, ml_forces)


    print("done")