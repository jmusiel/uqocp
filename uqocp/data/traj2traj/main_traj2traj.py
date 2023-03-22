import pickle
import os
from ase.io import Trajectory
from tqdm import tqdm
from finetuna.ml_potentials.finetuner_calc import FinetunerCalc
from finetuna.utils import compute_with_calc
import numpy as np
import argparse
from ase.optimize import BFGS
from finetuna.ml_potentials.ensemble_calc import EnsembleCalc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoints', nargs='+', help='<Required> List of paths to checkpoints for use, defaults to a preset list for an ensemble', required=False)
    parser.add_argument("-p","--picklefile", help='<Optional> Path to the pickle file for preloading tags, defaults to OC_20_val_data.pkl', required=False)
    parser.add_argument("-d", "--distributions", nargs='+', help='<Optional> Distributions in the pickle file to be sampled. Defaults to sampling all distributions.', required=False)
    parser.add_argument("-s", "--skip", help='<Optional> Whether to skip files that exist already, if passed then false ', action="store_true")
    parser.add_argument("-sd", "--subdir", help='<Optional> Which subdirectory to save results to, defaults to ocp_val')
    parser.add_argument("-ni", "--noinference", help='<Optional> Whether to perform inference, by default inference is performed and data is saved, if passed inference is not performed', action="store_true")
    parser.add_argument("-nr", "--norelax", help='<Optional> Whether to perform relaxations, by default relaxations are performed and relaxation data is saved, if passed relaxations are not performed', action="store_true")
    parser.add_argument("-ner", "--noensrelax", help='<Optional> Whether to perform a mean relaxation with the ensemble, by default ensemble relaxation is performed and relaxation data is saved, if passed ensemble relaxation is not performed', action="store_true")
    parser.add_argument("-m","--max", type=int, help='<Optional> Maximum number of systems to run inference on, defaults to running on all')
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
    else:
        checkpoints.append(None)
    print("checkpoints:")
    for c in checkpoints:
        print("\t" + str(c))

    # preload tags
    pkl_path = args.picklefile
    if pkl_path is None:
        # pkl_path=__file__[:__file__.rindex("/")] + "/OC_20_val_data.pkl"
        pkl_path="/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/data/OC_20_val_data.pkl"
    with open(os.path.join(pkl_path), "rb") as f:
        df = pickle.load(f)
    print("pickle path: " + str(pkl_path))

    # choose sample distributions, defaults to all
    distributions = args.distributions
    if distributions is None:
        distributions = list(set(df.distribution.to_list()))
    print("distributions: " + str(distributions))

    # choose whether to replace existing files, or skip inference on them
    skip = args.skip
    if skip is None:
        skip = False

    # get given subdir (defaults to ocp_val)
    subdir = args.subdir
    if subdir is None:
        subdir = "ocp_val"
    print("subdir: " + str(subdir))

    # check whether to also do an is2re relaxation (defaults to doing the is2re relaxation)
    norelax = args.norelax
    if norelax is None:
        norelax = True
    print("relaxing: " + str(not norelax))

    # check whether to also do an ensemble is2re relaxation (defaults to doing the ensemble is2re relaxation)
    noensrelax = args.noensrelax
    if noensrelax is None:
        noensrelax = True
    print("ensemble relaxing: " + str(not noensrelax))

    # check whether to do inference on given data (defaults to doing inference)
    noinference = args.noinference
    if noinference is None:
        noinference = True
    print("doing inference: " + str(not noinference))

    # check how many systems to run inference on (defaults to doing all)
    maxsystems = args.max
    print("max number of systems to run: " +str(maxsystems))


    # Now execute main script
    calcs_dict = {}
    ens_calcs_dict = {}
    for checkpoint_path in checkpoints:
        if checkpoint_path is not None:
            save_path = checkpoint_path.split("/")[-1].split(".")[0]
            save_path = os.path.join(subdir, save_path)
            os.makedirs(save_path, exist_ok=True)
            try:
                calcs_dict[save_path] = FinetunerCalc(checkpoint_path, mlp_params={
                    "cpu": False,
                    "optim": {
                        "batch_size":5
                    }
                })
                ens_calcs_dict[save_path] = calcs_dict[save_path]
            except:
                print("ERROR: hit exception initializing: " + str(checkpoint_path))
                quit()
        else:
            save_path = "dft"
            save_path = os.path.join(subdir, save_path)
            os.makedirs(save_path, exist_ok=True)
            calcs_dict[save_path] = None
            ens_save_path = save_path
    ens_calc = EnsembleCalc(ens_calcs_dict)

    for d in distributions:
        # trajids = df[df.distribution == d].random_id.tolist()
        trajids = df[df.distribution == d]
        j=0
        for i, row in tqdm(trajids.iterrows(), d):
            j = j+1
            if maxsystems is not None and j > maxsystems:
                break
            tid = row["random_id"]
            if "path" in row:
                traj = Trajectory(row["path"])
            else:
                traj = Trajectory("/home/jovyan/shared-datasets/OC20/trajs/val_02_01/"+tid+".traj")
            for save_path, calc in calcs_dict.items():
                if not noinference:
                    print(f"not noinference: {not noinference}")
                    writepath = save_path+"/"+tid+".traj"
                    if not skip or not os.path.isfile(writepath):
                        if calc is not None:
                            atoms_list = [x for x in compute_with_calc(traj, calc)]
                        else:
                            atoms_list = [x for x in traj]
                        writetraj = Trajectory(writepath, "w")
                        for x in atoms_list:
                            writetraj.write(atoms=x)

                if not norelax:
                    print(f"not norelax: {not norelax}")
                    writepath = save_path+"/"+tid+"_is2re.traj"
                    if not skip or not os.path.isfile(writepath):
                        if calc is not None:
                            structure = traj[0].copy()
                            structure.set_calculator(calc)
                            dyn = BFGS(structure, logfile=None, maxstep=0.2, trajectory=writepath)
                            dyn.run(fmax=0.03, steps=1000)

            if not noensrelax: # do the ensemble mean relaxation, save it to the dft folder and each individual inference to respective calc folders
                print(f"not noensrelax: {not noensrelax}")
                writepath = ens_save_path+"/"+tid+"_ens.traj"
                if not skip or not os.path.isfile(writepath):
                    structure = traj[0].copy()
                    structure.set_calculator(ens_calc)

                    trajsdict = {}
                    for save_path in ens_calcs_dict.keys():
                        trajsdict[save_path] = Trajectory(save_path+"/"+tid+"_ens.traj", "w")
                    def enswrite(image=structure,tdict=trajsdict):
                        for key, value in tdict.items():
                            value.write(atoms=image.calc.results["members"][key])

                    dyn = BFGS(structure, logfile=None, maxstep=0.2, trajectory=writepath)
                    dyn.attach(enswrite, interval=1)
                    dyn.run(fmax=0.03, steps=1000)




    print("done")