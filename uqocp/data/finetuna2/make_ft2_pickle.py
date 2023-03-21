import os
import pickle
import glob
from tqdm import tqdm
from ase.io import Trajectory
import pandas as pd


if __name__ == "__main__":
    pkl_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/OC_20_val_data.pkl"
    with open(os.path.join(pkl_path), "rb") as f:
        df = pickle.load(f)


    distributions = {
        "Sudheesh_mof_zeolite": "/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/MOFs_zeolites/Sudheesh_mof_zeolite/**/abinitio/oal_relaxation.traj",
        "Xiaoyan_zeolite": "/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/MOFs_zeolites/Xiaoyan_zeolite/**/abinitio/dft.traj",
        "BEEF_IPA": "/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/BEEF_IPA/**/vasp_inter_bfgs_relax.traj",
        "OC22": "/home/jovyan/shared-scratch/joe/repos/finetuna_2_data_private/data/oxides/OC22/**/abinitio/vasp_inter_bfgs_relax.traj", 
    }

    # todf = [["distribution", "random_id", "path"]]
    todf = {
        "distribution":[],
        "random_id":[],
        "path":[],
    }
    for d, search_path in distributions.items():
        trajpaths = [name for name in glob.glob(search_path, recursive=True)]
        for trajpath in tqdm(trajpaths, d):
            traj = Trajectory(trajpath)
            root = search_path.split("**")[0]
            stem = search_path.split("**")[1]
            tid = trajpath.replace(root,"").replace(stem,"").replace("/","_").replace(".","_")
            todf["distribution"].append(d)
            todf["random_id"].append(tid)
            todf["path"].append(trajpath)
            # for save_path, calc in calcs_dict.items():
            #     if not skip or not os.path.isfile(save_path+"/"+tid+".npy"):
            #         if calc is not None:
            #             ml_forces = np.array([x.get_forces() for x in compute_with_calc(traj, calc)])
            #         else:
            #             ml_forces = np.array([x.get_forces() for x in traj])
            #         np.save(save_path+"/"+tid, ml_forces)
    df = pd.DataFrame(todf)

    for i, row in tqdm(df.iterrows()):
        print(row["random_id"])
        if "path" in row:
            print(row["path"])
    
    with open(os.path.join("/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/finetuna2/ft2_id_data.pkl"), "wb") as f:
        pickle.dump(df, f)


    print("done")