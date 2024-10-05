import numpy as np
from ase.io import Trajectory, read
import os
from tqdm import tqdm
from pqdm.processes import pqdm
from uqocp.utils.error import get_force_diff
import pickle
from ocpmodels.datasets.lmdb_dataset import LmdbDataset
import glob
from uqocp.utils.load_data import sort_adsorbml_by_sid_and_fid, filter_adsorbml_by_split_type


# REF_ENERGIES_PKL_PATH = "/large_experiments/opencatalyst/electrocatalysis/relaxations/mapping/2021_02_07_splits_updated/mappings/final_ref_energies_02_07_2021.pkl"
# ADSORB_ML_REF_ENERGIES_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/ref_energies_updated_sysid.pkl"
# ADSORB_ML_RANDOM_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/random_metadata_by_sid.pkl"
# ADSORB_ML_HEURISTIC_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/heur_metadata_by_intsid.pkl"
# ADSORB_ML_RELEASE_PKL_PATH = "/large_experiments/opencatalyst/maaps/for_release/mappings/oc20dense_mapping.pkl"
# IS2RE_INITIAL_ENERGY_FOR_ADSORPTION_PKL_PATH = "/private/home/jmu/working/metadata/adsorption_energy_initial_pkl/adsorption_initial_energies.pkl"
REF_ENERGIES_PKL_PATH = "/home/jovyan/joe-uq/fair_s3_files/mappings/final_ref_energies_02_07_2021.pkl"
ADSORB_ML_REF_ENERGIES_PKL_PATH = "/home/jovyan/joe-uq/fair_s3_files/mappings/ref_energies_updated_sysid.pkl"
ADSORB_ML_RANDOM_PKL_PATH = "/home/jovyan/joe-uq/fair_s3_files/mappings/random_metadata_by_sid.pkl"
ADSORB_ML_HEURISTIC_PKL_PATH = "/home/jovyan/joe-uq/fair_s3_files/mappings/heur_metadata_by_intsid.pkl"
ADSORB_ML_RELEASE_PKL_PATH = "/home/jovyan/joe-uq/fair_s3_files/mappings/oc20dense_mapping.pkl"
IS2RE_INITIAL_ENERGY_FOR_ADSORPTION_PKL_PATH = "/home/jovyan/joe-uq/fair_s3_files/extracted/private/home/jmu/working/metadata/adsorption_energy_initial_pkl/adsorption_initial_energies.pkl"


def get_traj(random_id_str):
    # check oc20
    traj_path = f"/large_experiments/opencatalyst/electrocatalysis/relaxations/bulkadsorbate/restitch_jan2021_validation_all/random{random_id_str}.traj"
    if os.path.exists(traj_path):
        return Trajectory(traj_path)
    traj_path = f"/large_experiments/opencatalyst/electrocatalysis/relaxations/bulkadsorbate/restitch_jan2021/random{random_id_str}.traj"
    if os.path.exists(traj_path):
        return Trajectory(traj_path)
    traj_path = f"/large_experiments/opencatalyst/electrocatalysis/relaxations/bulkadsorbate/restitch_jan2021_test_all/random{random_id_str}.traj"
    if os.path.exists(traj_path):
        return Trajectory(traj_path)
    raise ValueError(f"random id {random_id_str} doesn't exist")

def get_oc22traj(random_id_str):
    # oc22
    if "oc22_ref" not in globals():
        with open("/private/home/jmu/working/metadata/ocp_mapping/oc22_metadata.pkl", "rb") as f:
            oc22_ref = pickle.load(f)
    traj_id = oc22_ref[int(random_id_str)]["traj_id"]
    traj_path = f"/large_experiments/opencatalyst/data/oc22/2022_06_16/raw_trajs_untrimmed_s2ef_train_val/{traj_id}.traj"
    if os.path.exists(traj_path):
        return Trajectory(traj_path)
    raise ValueError(f"random id {random_id_str} doesn't exist")


def get_forces_from_traj(frames_from_traj):
    current_id, frames_list, dataset = frames_from_traj
    forces_list = []
    if dataset == "oc22":
        current_traj = get_oc22traj(current_id)
    else:
        current_traj = get_traj(current_id)
    frame = current_traj[0]
    if len(frame.constraints) > 0:
        constraints = frame.constraints[0].index
    else:
        constraints = []
    unconstrained_atoms = [i for i in range(len(frame)) if i not in constraints]
    for f in frames_list:
        frame = current_traj[f]
        f_forces = frame.get_forces()[unconstrained_atoms]
        forces_list.extend(f_forces)
    return current_id, forces_list


def save_ensemble_force_results(output_filepath, results_filepath_list, dataset="oc20", debug=False):
    if dataset not in ["oc20", "oc22"]:
        raise ValueError(f"invalid dataset: {dataset}")

    ids_dict = {}
    forces_dict = {}
    for result_filepath in results_filepath_list:
        key = result_filepath.split("/")[-2]
        forces_list = []
        with np.load(result_filepath) as data:
            ids_dict[key] = data["ids"]
            forces_dict[key] = data["forces"]

    given_id_order = []
    all_trajs = []
    current_id = ""
    for id_str in tqdm(ids_dict[key], "accumulating all trajs forces"):
        i, f = id_str.split("_")
        if current_id == i:
            all_trajs[-1][1].append(int(f))
        else:
            current_id = i
            given_id_order.append(current_id)
            all_trajs.append([current_id, [], dataset])
            all_trajs[-1][1].append(int(f)) 
    if debug:
        all_trajs = all_trajs[:10] # debug
    result = pqdm(all_trajs, get_forces_from_traj, n_jobs=64)
    forces_list = []
    dft_id_order = []
    for current_id, ls in result:
        dft_id_order.append(current_id)
        forces_list.extend(ls)
    forces_dict["dft"] = np.array(forces_list)
    for given_id, dft_id in zip(given_id_order, dft_id_order):
        assert given_id == dft_id
    np.savez(output_filepath, **forces_dict)

def save_ensemble_energy_results(output_filepath, results_filepath_list, lmdb_filepath, need_ref_energies_for_created_lmdb=False, debug=False, domain=None):
    if need_ref_energies_for_created_lmdb:
        with open(REF_ENERGIES_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)

    energy_dict = {}
    ids_dict = {}
    for result_filepath in results_filepath_list:
        key = result_filepath.split("/")[-2]
        with np.load(result_filepath) as data:
            ids_dict[key] = data["ids"]
            energy_dict[key] = data["energy"]

    given_id_order = ids_dict[key].tolist()

    energy_list = []
    dft_id_order = []

    ldata = LmdbDataset(config={"src":lmdb_filepath})
    for i, d in tqdm(enumerate(ldata), f"enumerating lmdb ({lmdb_filepath.split('data/')[-1]})", total=len(ldata)):
        energy = d["y"]
        try:
            dft_id = f"{d['sid'].numpy()[0]}_{d['fid'].numpy()[0]}"
        except AttributeError as e:
            dft_id = f"{d['sid']}_{d['fid']}"
        dft_id_order.append(dft_id)
        if need_ref_energies_for_created_lmdb:
            energy = energy - ref_energies[f"random{dft_id.split('_')[0]}"]
        energy_list.append(energy)
        if debug:
            if not given_id_order.count(dft_id) == 1:
                print(f"{dft_id} in given ids {given_id_order.count(dft_id)} times")
            if i > 1000:
                break

    if debug:
        debug_order = []
        backup_dft_id_order = [i for i in dft_id_order]
        for given_id in tqdm(given_id_order, "debug extraction"):
            if str(given_id) in dft_id_order:
                debug_order.append(given_id)
                backup_dft_id_order.remove(given_id)
        print(f"len given: {len(debug_order)}, len dft: {len(dft_id_order)}, len backup: {len(backup_dft_id_order)}, backup: {backup_dft_id_order}")
        given_id_order = debug_order

    dft_id_order, energy_list = zip(*sorted(zip(dft_id_order, energy_list)))
    energy_dict["dft"] = np.array(energy_list)
    for given_id, dft_id in zip(given_id_order, dft_id_order):
        if not given_id == dft_id:
            print(f"given: {given_id}, dft: {dft_id}")
        assert given_id == dft_id
    np.savez(output_filepath, **energy_dict)


def get_mean_ensemble_err(ensemble_filepath, keys=None, diff_type="l2"):
    forces_list = []
    with np.load(ensemble_filepath) as data:
        dft_forces = data["dft"]
        for key in data:
            if not key == "dft":
                if keys is None or _in_keys_bool(key, keys):
                    forces_list.append(data[key])
    forces_arr = np.array(forces_list)
    mean_forces = np.mean(forces_arr, axis=0)
    err = get_force_diff(dft_forces, mean_forces, diff_type=diff_type)
    return err

def get_mean_ensemble_unc(ensemble_filepath, keys=None, diff_type="l2"):
    forces_list = []
    with np.load(ensemble_filepath) as data:
        dft_forces = data["dft"]
        for key in data:
            if not key == "dft":
                if keys is None or _in_keys_bool(key, keys):
                    forces_list.append(data[key])
    forces_arr = np.array(forces_list)
    mean_forces = np.mean(forces_arr, axis=0)
    
    differences = []
    for forces in tqdm(forces_list, "getting unc"):
        diff = get_force_diff(mean_forces, forces, diff_type=diff_type)
        differences.append(diff)
    diffs_arr = np.array(differences)
    n = diffs_arr.shape[0]
    unc = np.sqrt(np.sum(np.square(diffs_arr), axis=0)/n)
    return unc


def get_per_traj_ensemble_unc_and_err(traj_dir, unc_type="max_l2", err_type="energy", debug=False, use_reference=True, dft_path=None):
    if use_reference:
        with open(REF_ENERGIES_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)

    pred_traj_path_list = glob.glob(os.path.join(traj_dir, "*_0.traj"))

    inputs = []
    for pred_traj_path in tqdm(pred_traj_path_list, "making path_utype_etype inputs"):
        if use_reference:
            ref_e = ref_energies[f"random{pred_traj_path.split('/')[-1].split('_0')[0]}"]
        else:
            ref_e = 0
        path_utype_etype = pred_traj_path, unc_type, err_type, ref_e, dft_path
        inputs.append(path_utype_etype)

    if debug:
        result = [_helper_get_per_traj_ensemble_unc_and_err(path_utype_etype) for path_utype_etype in tqdm(inputs, "getting unc and err")]
    else:
        result = pqdm(inputs, _helper_get_per_traj_ensemble_unc_and_err, n_jobs=64)

    unc_list = []
    err_list = []
    for unc, err in tqdm(result, "creating final lists"):
        if unc is not None and err is not None:
            unc_list.append(unc)
            err_list.append(err)
    return np.array(unc_list), np.array(err_list)

def _helper_get_per_traj_ensemble_unc_and_err(path_utype_etype):
    pred_traj_path, unc_type, err_type, ref_e, dft_path = path_utype_etype

    sid = pred_traj_path.split("/")[-1].split("_0")[0]
    pred_traj = Trajectory(pred_traj_path)
    if dft_path is None:
        target_traj = get_traj(sid)
    else:
        try:
            target_traj = [read(os.path.join(dft_path, sid, "vasprun.xml"))]
        except:
            return None, None

    if err_type == "energy":
        prediction = pred_traj[-1].get_potential_energy()
        target = target_traj[-1].get_potential_energy() - ref_e
        err = get_force_diff(prediction, target, "energy")
    elif err_type in ["l2", "l1", "mag", "cos"]:
        prediction = pred_traj[-1].get_forces()
        target = target_traj[-1].get_forces()
        err = np.mean(get_force_diff(prediction, target, err_type))
    else:
        raise ValueError(f"invalid error type: {err_type}")

    if unc_type == "max_energy":
        traj_unc_list = [a.info["energy_unc"] for a in pred_traj]
        unc = np.max(traj_unc_list)
    elif unc_type == "mean_energy":
        traj_unc_list = [a.info["energy_unc"] for a in pred_traj]
        unc = np.mean(traj_unc_list)
    elif unc_type == "max_l2":
        traj_unc_list = [a.info["force_unc"] for a in pred_traj]
        unc = np.max(traj_unc_list)
    elif unc_type == "mean_l2":
        traj_unc_list = [a.info["force_unc"] for a in pred_traj]
        unc = np.mean(traj_unc_list)
    else:
        raise ValueError(f"invalid uncertainty type: {unc_type}")
    
    return unc, err

def get_adsorbml_ensemble_uncertainty_random_heuristic_hack(
    unc_path_list: list[str],
    ml_path: str,
    dft_path: str,
    unc_type="max_l2",
    err_type="ensemble_energy",
    debug=False,
    use_reference=True,
):
    heuristic_unc_path_list = []
    random_unc_path_list = []
    release_unc_path_list = []
    for unc_path in unc_path_list:
        if unc_path.count("s2ef_predictions.npz") == 1:
            if "random" in unc_path.split("/")[-5]:
                random_unc_path_list.append(unc_path)
            elif "heuristic" in unc_path.split("/")[-5]:
                heuristic_unc_path_list.append(unc_path)
            elif "release" in unc_path:
                release_unc_path_list.append(unc_path)
            else:
                raise ValueError(f"invalid path, no random or heurstic in [-5] directory: {unc_path}")
        elif unc_path.count("s2ef_predictions.npz") == 2:
            path1, path2, _ = unc_path.split("s2ef_predictions.npz")
            for path in [path1, path2]:
                if "random" in path.split("/")[-5]:
                    random_unc_path_list.append(path + "s2ef_predictions.npz")
                elif "heuristic" in path.split("/")[-5]:
                    heuristic_unc_path_list.append(path + "s2ef_predictions.npz")
                else:
                    raise ValueError(f"invalid path, no random or heurstic in [-5] directory: {unc_path}")
        else:
            raise ValueError(f"invalid path, no s2ef_predictions.npz in path: {unc_path}")
        
    unc_list = []
    err_list = []
    if len(heuristic_unc_path_list) == len(unc_path_list):
        unc, err = get_adsorbml_ensemble_uncertainty(
            unc_path_list=heuristic_unc_path_list,
            ml_path=ml_path,
            dft_path=dft_path,
            placement_strategy="heuristic",
            unc_type=unc_type,
            err_type=err_type,
            debug=debug,
            use_reference=use_reference,
        )
        unc_list.append(unc)
        err_list.append(err)
    elif len(heuristic_unc_path_list) == 0:
        pass
    else:
        raise ValueError(f"invalid number of heuristic paths: {len(heuristic_unc_path_list)}")
    
    if len(random_unc_path_list) == len(unc_path_list):
        unc, err = get_adsorbml_ensemble_uncertainty(
            unc_path_list=random_unc_path_list,
            ml_path=ml_path,
            dft_path=dft_path,
            placement_strategy="random",
            unc_type=unc_type,
            err_type=err_type,
            debug=debug,
            use_reference=use_reference,
        )
        unc_list.append(unc)
        err_list.append(err)
    elif len(random_unc_path_list) == 0:
        pass
    else:
        raise ValueError(f"invalid number of random paths: {len(random_unc_path_list)}")
    
    if len(release_unc_path_list) == len(unc_path_list):
        unc, err = get_adsorbml_ensemble_uncertainty(
            unc_path_list=release_unc_path_list,
            ml_path=ml_path,
            dft_path=dft_path,
            placement_strategy="release",
            unc_type=unc_type,
            err_type=err_type,
            debug=debug,
            use_reference=use_reference,
        )
        unc_list.append(unc)
        err_list.append(err)
    
    unc = np.concatenate(unc_list)
    err = np.concatenate(err_list)
    return unc, err
            

def get_adsorbml_ensemble_uncertainty(
        unc_path_list: list[str],
        ml_path: str,
        dft_path: str,
        placement_strategy: str, # random or heuristic
        unc_type="max_l2",
        err_type="ensemble_energy",
        calculate_adsorption_error=False,
        debug=False,
        use_reference=True,
    ):

    ensemble_data = {
        "ids": None,
        "chunk_idx": None,
        "energy": [],
        "forces": [],
        "mean_energy": None,
        "mean_forces": None,
        "sid": [],
        "fid": [],
    }
    # enumerate indices
    with np.load(unc_path_list[0]) as data:
        ids_arr = data["ids"]
        frame_chunk_idx = data["chunk_idx"]
        for i, ids in tqdm(enumerate(ids_arr), "getting ids from first npz file", total=len(ids_arr)):
            sid, fid = ids.split("_")
            ensemble_data["sid"].append(int(sid))
            ensemble_data["fid"].append(int(fid))
    ensemble_data["sid"] = np.array(ensemble_data["sid"])
    ensemble_data["fid"] = np.array(ensemble_data["fid"])

    # get indices sorted by sid first and fid second get sorted system chunk idx
    sorted_indices = np.lexsort((ensemble_data["fid"], ensemble_data["sid"]), axis=0)
    sorted_sids = ensemble_data["sid"][sorted_indices]
    prev_sid = sorted_sids[0]
    system_chunk_idx = []
    for i, sid in tqdm(enumerate(sorted_sids), "getting system_chunk_idx", total=len(sorted_sids)):
        if sid != prev_sid:
            system_chunk_idx.append(i)
        prev_sid = sid
    system_chunk_idx = np.array(system_chunk_idx)
    unique_sid_order = list(dict.fromkeys(sorted_sids))

    # accumumlate results
    for unc_path in tqdm(unc_path_list, "accumulating prediction results"):
        with np.load(unc_path) as data:
            sorted_energy = data["energy"][sorted_indices]
            if calculate_adsorption_error:
                j = 0
                system_chunk_idx_with_0 = np.append(np.append([0], system_chunk_idx), [len(sorted_energy)+1])
                sorted_energy_copy = np.zeros_like(sorted_energy)
                for i, e in enumerate(sorted_energy):
                    if i == system_chunk_idx_with_0[j]:
                        j += 1
                    sorted_energy_copy[i] = sorted_energy[i] - sorted_energy[system_chunk_idx_with_0[j-1]]
                sorted_energy = sorted_energy_copy
            ensemble_data["energy"].append(sorted_energy)
            forces = np.split(data["forces"], frame_chunk_idx)
            forces = [forces[i] for i in sorted_indices]
            new_frame_chunk_idx = np.cumsum([len(f) for f in forces])[:-1]
            forces = np.concatenate(forces, axis=0)
            ensemble_data["forces"].append(forces)
    ensemble_data["mean_energy"] = np.mean(ensemble_data["energy"], axis=0)
    ensemble_data["mean_forces"] = np.mean(ensemble_data["forces"], axis=0)
    frame_chunk_idx = new_frame_chunk_idx
        
    # check which unertainty type to use
    unc_mod, unc_type = unc_type.split("_")
    if unc_type == "energy":
        predictions_list_for_unc = ensemble_data["energy"]
        mean_predictions_for_unc = ensemble_data["mean_energy"]
    elif unc_type in ["l2", "l1", "mag", "cos"]:
        predictions_list_for_unc = ensemble_data["forces"]
        mean_predictions_for_unc = ensemble_data["mean_forces"]
    else:
        raise ValueError(f"invalid uncertainty type: {unc_mod}_{unc_type}")

    # get uncertainty
    differences = []
    for predictions_for_unc in tqdm(predictions_list_for_unc, "getting unc"):
        diff = get_force_diff(mean_predictions_for_unc, predictions_for_unc, diff_type=unc_type)
        differences.append(diff)
    diffs_arr = np.array(differences)
    n = diffs_arr.shape[0]
    per_frame_unc = np.sqrt(np.sum(np.square(diffs_arr), axis=0)/n)

    # if forces, split uncertainty into per frame
    if unc_type in ["l2", "l1", "mag", "cos"]:
        per_frame_unc = np.split(per_frame_unc, frame_chunk_idx)

    # get uncertainty on a per system basis
    per_sys_unc = np.split(per_frame_unc, system_chunk_idx)
    if unc_mod == "max":
        unc = np.array([np.max([np.max(frame) for frame in system]) for system in per_sys_unc])
    elif unc_mod == "mean":
        unc = np.array([np.mean([np.mean(frame) for frame in system]) for system in per_sys_unc])
    elif unc_mod == "first":
        unc = np.array([np.mean(u[0]) for u in per_sys_unc])
    elif unc_mod == "last":
        unc = np.array([np.mean(u[-1]) for u in per_sys_unc])

    # check which error type to use
    err_mod, err_type = err_type.split("_")
    # get predictions to go with targets for error
    if err_mod == "ensemble":
        if err_type == "energy":
            predictions_for_err = ensemble_data["mean_energy"]
            per_sys_predictions_for_err = np.split(predictions_for_err, system_chunk_idx)
        elif err_type in ["l2", "l1", "mag", "cos"]:
            predictions_for_err = ensemble_data["mean_forces"]
            per_frame_predictions_for_err = np.split(predictions_for_err, frame_chunk_idx)
            per_sys_predictions_for_err = np.split(per_frame_predictions_for_err, system_chunk_idx)
    else:
        per_sys_predictions_for_err=None

    err = adsorbml_error_helper(
        unique_sid_order=unique_sid_order,
        ml_path=ml_path,
        dft_path=dft_path,
        placement_strategy=placement_strategy,
        err_type=err_type,
        per_sys_predictions_for_err=per_sys_predictions_for_err,
        calculate_adsorption_error=calculate_adsorption_error,
        use_reference=use_reference,
        debug=debug,
    )
    
    # continue handling some errors not existing because some vasprun.xml files were not parsable
    _err = []
    _unc = []
    for _u, _e in zip (unc, err):
        if _e is not None and _u is not None:
            _err.append(_e)
            _unc.append(_u)
    err = np.array(_err)
    unc = np.array(_unc)

    return unc, err

def adsorbml_error_helper(
    unique_sid_order: list[int],
    ml_path: str,
    dft_path: str,
    placement_strategy: str,
    err_type: str,
    per_sys_predictions_for_err: list=None,
    calculate_adsorption_error=False,
    use_reference=True,
    debug=False,
):
    if use_reference and not placement_strategy == "is2re":
        with open(ADSORB_ML_REF_ENERGIES_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)

    if placement_strategy == "random":
        with open(ADSORB_ML_RANDOM_PKL_PATH, "rb") as f:
            mapping_dict = pickle.load(f)
    elif placement_strategy == "heuristic":
        with open(ADSORB_ML_HEURISTIC_PKL_PATH, "rb") as f:
            mapping_dict = pickle.load(f)
    elif placement_strategy == "release":
        with open(ADSORB_ML_RELEASE_PKL_PATH, "rb") as f:
            mapping_dict = pickle.load(f)
    elif placement_strategy == "is2re":
        with open(REF_ENERGIES_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)
            mapping_dict = {sid:sid for sid in unique_sid_order}
    else:
        raise ValueError(f"invalid placement strategy: {placement_strategy}")

    if calculate_adsorption_error:
        if not placement_strategy == "is2re":
            raise ValueError(f"can only calculate adsorption error for is2re, not {placement_strategy}")
        with open(IS2RE_INITIAL_ENERGY_FOR_ADSORPTION_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)
    
    # get inputs for error targets pqdm
    target_inputs = []
    for sid in unique_sid_order:
        mapping = mapping_dict[sid]
        if use_reference:
            if placement_strategy == "release":
                ref_e = ref_energies[mapping['system_id']]
            elif placement_strategy == "is2re":
                ref_e = ref_energies[f"random{mapping}"]
            else:
                ref_e = ref_energies[mapping[0]]
        else:
            ref_e = 0
        if placement_strategy == "release":
            dft_mapping_path = os.path.join(dft_path, f"{mapping['system_id']}_{mapping['config_id']}.traj")
        elif placement_strategy == "is2re":
            dft_mapping_path = os.path.join(dft_path, str(mapping), "vasprun.xml")
            if not os.path.exists(dft_mapping_path):
                continue
        else:
            dft_mapping_path = os.path.join(dft_path, f"{mapping[0]}_{mapping[1]}.traj")
        target_inputs.append((dft_mapping_path, ref_e, err_type))
    
    # get targets
    if debug:
        targets = []
        debug_args = []
        for i, inp in tqdm(enumerate(target_inputs), "getting err targets (DEBUG)", total=len(target_inputs)):
            if os.path.exists(inp[0]):
                try:
                    targets.append(_get_adsorbml_ensemble_error_target_helper(inp))
                except Exception as exc:
                    targets.append(exc)
                debug_args.append(i)
            if i > 512:
                break
        debug_args = np.array(debug_args)
    else:
        targets = pqdm(target_inputs, _get_adsorbml_ensemble_error_target_helper, n_jobs=1, desc="getting err targets")

    # get predictions to go with targets for error
    if per_sys_predictions_for_err is None:
        # prep inputs for pqdm
        pred_inputs = []
        for sid in unique_sid_order:
            if placement_strategy == "is2re":
                pred_inputs.append((os.path.join(ml_path, f"{sid}.traj"), err_type, calculate_adsorption_error))
            else:
                if os.path.exists(os.path.join(ml_path, placement_strategy, f"{sid}.traj")):
                    pred_inputs.append((os.path.join(ml_path, placement_strategy, f"{sid}.traj"), err_type, calculate_adsorption_error))
                else:
                    raise ValueError(f"sid {sid} not in {placement_strategy} ml path {os.path.join(ml_path, placement_strategy, f'{sid}.traj')}")
        # execute pqdm to retrieve data from .traj files
        if debug:
            per_sys_final_predictions_for_err = []
            for i, inp in tqdm(enumerate(pred_inputs), "getting err pred (DEBUG)", total=len(pred_inputs)):
                per_sys_final_predictions_for_err.append(_get_adsorbml_ensemble_error_prediction_helper(inp))
                if i > 512:
                    break
        else:
            per_sys_final_predictions_for_err = pqdm(pred_inputs, _get_adsorbml_ensemble_error_prediction_helper, n_jobs=64, desc="getting err pred")
    else:
        if calculate_adsorption_error:
            per_sys_final_predictions_for_err = [s[-1]-s[0] for s in per_sys_predictions_for_err]
        else:
            per_sys_final_predictions_for_err = [s[-1] for s in per_sys_predictions_for_err]
    
    if debug:
        per_sys_final_predictions_for_err = [per_sys_final_predictions_for_err[i] for i in debug_args]

    # handle some vasprun.xml files not existing or not being parsable
    exceptions_pred = [exc for exc in per_sys_final_predictions_for_err if isinstance(exc, Exception)]
    if len(exceptions_pred) > 0:
        print(f"exception in predictions for errors: ({len(exceptions_pred)} exceptions)\n\t{exceptions_pred[:min(len(exceptions_pred), 10)]}", flush=True)
    exceptions_targ = [exc for exc in targets if isinstance(exc, Exception)]
    if len(exceptions_targ) > 0:
        print(f"exception in targets for errors: ({len(exceptions_targ)} exceptions)\n\t{exceptions_targ[:min(len(exceptions_targ), 10)]}", flush=True)
    existing_values = [True for i in targets]
    if len(exceptions_targ) > 0 or len(exceptions_pred) > 0:
        _predictions = []
        _targets = []
        existing_values = []
        for pred, targ in zip(per_sys_final_predictions_for_err, targets):
            if not isinstance(pred, Exception) and not isinstance(targ, Exception):
                _predictions.append(pred)
                _targets.append(targ)
                existing_values.append(True)
            else:
                existing_values.append(False)
        per_sys_final_predictions_for_err = np.array(_predictions)
        targets = np.array(_targets)

    # finally get error
    if err_type == "energy":
        per_sys_final_predictions_for_err = np.array(per_sys_final_predictions_for_err)
        targets = np.array(targets)
        err = get_force_diff(targets, per_sys_final_predictions_for_err, diff_type=err_type)
    elif err_type in ["l2", "l1", "mag", "cos"]:
        final_chunk_idx = np.cumsum([len(f) for f in per_sys_final_predictions_for_err])[:-1]
        per_sys_final_predictions_for_err = np.concatenate(per_sys_final_predictions_for_err, axis=0)
        targets = np.concatenate(targets, axis=0)
        err = get_force_diff(targets, per_sys_final_predictions_for_err, diff_type=err_type)
        err = [np.mean(e) for e in np.split(err, final_chunk_idx)]

    # continue handling some vasprun.xml files not existing or not being parsable
    _err = []
    i = 0
    mean_err = 0
    for existence in existing_values:
        if existence:
            mean_err += err[i]/len(err)
            _err.append(err[i])
            i += 1
        else:
            _err.append(None)
    err = np.array(_err)
    print(f"after collection err, mean err: {mean_err}")
    return err

def _get_adsorbml_ensemble_error_target_helper(dft_path_tuple):
    dft_mapping_path, ref_e, err_type = dft_path_tuple
    try:
        if dft_mapping_path.endswith(".traj"):
            atoms = Trajectory(dft_mapping_path)[-1]
        else:
            atoms = read(dft_mapping_path)
        if err_type == "energy":
            target = atoms.get_potential_energy() - ref_e
        elif err_type in ["l2", "l1", "mag", "cos"]:
            target = atoms.get_forces()
            target = target[[i for i, a in enumerate(atoms) if i not in atoms.constraints[0].index]]
        else:
            raise ValueError(f"invalid error type: {err_type}")
        return target
    except Exception as e:
        return e

def _get_adsorbml_ensemble_error_prediction_helper(ml_path_tuple):
    ml_path, err_type, calculate_adsorption_error = ml_path_tuple
    traj = Trajectory(ml_path)
    atoms = traj[-1]
    if err_type == "energy":
        prediction = atoms.get_potential_energy()
    elif err_type in ["l2", "l1", "mag", "cos"]:
        prediction = atoms.get_forces()
        prediction = prediction[[i for i, a in enumerate(atoms) if i not in atoms.constraints[0].index]]
    else:
        raise ValueError(f"invalid error type: {err_type}")
    if calculate_adsorption_error:
        prediction = prediction - traj[0].get_potential_energy()
    return prediction

def get_adsorbml_residual_head_uncertainty(
    unc_path: str,
    ml_path: str,
    dft_path: str,
    placement_strategy: str, # random or heuristic
    unc_type="max_energy",
    err_type="traj_energy",
    split_type=None,
    debug=False,
    use_reference=True,
    ):
    
    if placement_strategy == "random":
        with open(ADSORB_ML_RANDOM_PKL_PATH, "rb") as f:
            mapping_dict = pickle.load(f)
    elif placement_strategy == "heuristic":
        with open(ADSORB_ML_HEURISTIC_PKL_PATH, "rb") as f:
            mapping_dict = pickle.load(f)
    elif placement_strategy == "release":
        with open(ADSORB_ML_RELEASE_PKL_PATH, "rb") as f:
            mapping_dict = pickle.load(f)
    else:
        raise ValueError(f"invalid placement strategy: {placement_strategy}")

    # check which unertainty type to use
    unc_mod, unc_type = unc_type.split("_")
    if not unc_type == "energy":
        raise ValueError(f"invalid uncertainty type: {unc_type}, adsorbml residual only supports energy")
    # check which error type to use
    err_mod, err_type = err_type.split("_")
    if not err_mod == "traj":
        raise ValueError(f"invalid error type: {err_mod}, adsorbml residual only supports traj")
    
    # extract sorted residuals predictions for uncertainty using adsorbml functions
    unique_sid_order, per_system_data = sort_adsorbml_by_sid_and_fid(unc_path, per_frame_keys=["residual"])
    if split_type is not None:
        unique_sid_order, per_system_data = filter_adsorbml_by_split_type(unique_sid_order, per_system_data, split_type, mapping_dict)
    
    # get the uncertainty
    if unc_mod == "max":
        unc = np.array([np.max(np.abs(res)) for res in per_system_data["residual"]])
    elif unc_mod == "mean":
        unc = np.array([np.mean(np.abs(res)) for res in per_system_data["residual"]])
    elif unc_mod == "first":
        unc = np.array([np.abs(res[0]) for res in per_system_data["residual"]])
    elif unc_mod == "last":
        unc = np.array([np.abs(res[-1]) for res in per_system_data["residual"]])
    else:
        raise ValueError(f"invalid uncertainty modifier: {unc_mod}")
    
    # get the error
    err = adsorbml_error_helper(
        unique_sid_order=unique_sid_order,
        ml_path=ml_path,
        dft_path=dft_path,
        placement_strategy=placement_strategy,
        err_type=err_type,
        per_sys_predictions_for_err=None,
        use_reference=use_reference,
        debug=debug,
    )

    return unc, err

def get_adsorbml_residual_head_uncertainty_placement_and_id_ood_hack(
    all_unc_path_list: list[str],
    ml_path: str,
    dft_path: str,
    split_type: list[str],
    unc_type="max_energy",
    err_type="traj_energy",
    debug=False,
    use_reference=True,
    ):
    if type(all_unc_path_list) == str:
        all_unc_path_list = [all_unc_path_list]

    all_unc = []
    all_err = []
    for unc_path in all_unc_path_list:
        if "heuristic" in unc_path and "random" in unc_path:
            raise ValueError(f"invalid uncertainty path: {unc_path}, found both heuristic and random")
        elif "heuristic" in unc_path:
            place_strat = "heuristic"
        elif "random" in unc_path:
            place_strat = "random"
        elif "release" in unc_path:
            place_strat = "release"
        else:
            raise ValueError(f"invalid uncertainty path: {unc_path}, could not find heuristic or random or release")
        unc, err = get_adsorbml_residual_head_uncertainty(
            unc_path=unc_path,
            ml_path=ml_path,
            dft_path=dft_path,
            placement_strategy=place_strat,
            unc_type=unc_type,
            err_type=err_type,
            split_type=split_type,
            debug=debug,
            use_reference=use_reference,
        )
        all_unc.append(unc)
        all_err.append(err)

    return np.concatenate(all_unc), np.concatenate(all_err)

def get_meta_npz_results(npz_filepath):
    with np.load(npz_filepath) as data:
        unc = data["unc"]
        err = data["err"]
    return unc, err

def get_unc_err_is2re_residual_head(
    unc_path: str,
    ml_path: str,
    dft_path: str,
    unc_type="max_energy",
    err_type="traj_energy",
    use_reference: bool=True,
):
    with np.load(unc_path) as data:
        unc_dict = dict(data)

    fids = []
    sids = []
    for sid_fid in unc_dict["ids"]:
        sid, fid = sid_fid.split("_")
        fids.append(int(fid))
        sids.append(int(sid))
    sids = np.array(sids)
    fids = np.array(fids)

    sorted_indices = np.lexsort((fids, sids), axis=0)
    sorted_sids = sids[sorted_indices]
    prev_sid = sorted_sids[0]
    system_chunk_idx = []
    for i, sid in tqdm(enumerate(sorted_sids), "getting system_chunk_idx", total=len(sorted_sids)):
        if sid != prev_sid:
            system_chunk_idx.append(i)
        prev_sid = sid
    system_chunk_idx = np.array(system_chunk_idx)
    unique_sid_order = list(dict.fromkeys(sorted_sids))

    sorted_residual = unc_dict["residual"][sorted_indices]  
    per_sys_unc = np.split(sorted_residual, system_chunk_idx)
    unc = []
    for system in per_sys_unc:
        if "max" in unc_type:
            unc.append(np.max(system))
        elif "mean" in unc_type:
            unc.append(np.mean(system))
        elif "first" in unc_type:
            unc.append(system[0])
        elif "last" in unc_type:
            unc.append(system[-1])
        else:
            raise ValueError(f"invalid uncertainty type: {unc_type}")
    unc = np.array(unc)

    err = adsorbml_error_helper(
        unique_sid_order=unique_sid_order,
        ml_path=ml_path,
        dft_path=dft_path,
        placement_strategy="is2re",
        err_type=err_type,
        per_sys_predictions_for_err=None,
        calculate_adsorption_error=False,
        use_reference=True,
        debug=False,
    )

    assert len(unc) == len(err)
    temp_unc = []
    temp_err = []
    for i, (u, e) in enumerate(zip(unc, err)):
        if u is not None and e is not None:
            temp_unc.append(u)
            temp_err.append(e)
    unc = np.array(temp_unc)
    err = np.array(temp_err)

    return unc, err

def _in_keys_bool(current_key_string, given_keys_substrings):
    """
    check if any of the given_keys_substrings appear in current_key_string, if yes return True
    :param current_key_string: The current key (full string) from the .npz file.
    :param given_keys_substrings: The list of substrings given as the 'keys' parameter to be checked against.
    """
    result = False
    for substring in given_keys_substrings:
        if substring in current_key_string:
            result = True
    return result

if __name__ == "__main__":
    # unc, err = get_per_traj_ensemble_unc_and_err("/private/home/jmu/storage/trajectory_results/test/test_trajs", unc_type="mean_energy", err_type="energy", debug=True, dft_path="/private/home/jmu/storage/trajectory_results/oc20data_oc20models_top4/is2re_val_ood/vasp_results/first10k/vasp_dirs")

    # unc, err = get_per_traj_ensemble_unc_and_err("/private/home/jmu/storage/trajectory_results/oc20data_oc20models_top4/is2re_val_ood/trajs", unc_type="max_l2", debug=True, dft_path="/private/home/jmu/storage/trajectory_results/oc20data_oc20models_top4/is2re_val_ood/vasp_results/first10k/vasp_dirs")

    # unc, err = get_adsorbml_ensemble_uncertainty(
    #     unc_path_list=[
    #         "/private/home/jmu/storage/inference/latent_adsorbml/adsorbml_escn_test_heuristic_150cutoff/escn_l4_m2_lay12_2M_s2ef/results/m-escn_l4_m2_lay12_2M_s2ef_d-test_heuristic/s2ef_predictions.npz",
    #         "/private/home/jmu/storage/inference/latent_adsorbml/adsorbml_escn_test_heuristic_150cutoff/escn_l6_m2_lay12_2M_s2ef/results/m-escn_l6_m2_lay12_2M_s2ef_d-test_heuristic/s2ef_predictions.npz"
    #     ],
    #     ml_path="/large_experiments/opencatalyst/maaps/escn/ml_trajs",
    #     dft_path="/large_experiments/opencatalyst/maaps/escn/dft/sp/clean_outputs",
    #     placement_strategy="heuristic",
    #     unc_type="max_l2",
    #     err_type="traj_l2",
    #     debug=False,
    #     use_reference=True,
    # )

    # unc, err = get_adsorbml_residual_head_uncertainty(
    #     unc_path="/private/home/jmu/storage/inference/adsorbml_residual_head/residual_ensemble_10h_escn_500k_steps/adsorbml_escn_val_heuristic_150cutoff/best_checkpoint/results/2023-07-10-11-25-04-m-best_checkpoint_d-val_heuristic/s2ef_predictions.npz",
    #     ml_path="/large_experiments/opencatalyst/maaps/escn/ml_trajs",
    #     dft_path="/large_experiments/opencatalyst/maaps/escn/dft/sp/clean_outputs",
    #     placement_strategy="heuristic",
    #     unc_type="max_energy",
    #     err_type="traj_energy",
    #     split_type=["id"],
    #     debug=False,
    #     use_reference=True,
    # )

    # unc, err = get_adsorbml_ensemble_uncertainty(
    #     unc_path_list=[
    #         "/private/home/jmu/storage/inference/equiformer_ensemble/lmax_and_layer_norm/2_epochs/eq2_is2re_traj_inf_ood_both/eq2_31M_ec4_allmd/results/2023-08-14-17-23-28-predict_eq2_is2re_traj_inf_ood_both_eq2_31M_ec4_allmd/s2ef_predictions.npz",
    #         "/private/home/jmu/storage/inference/equiformer_ensemble/lmax_and_layer_norm/2_epochs/eq2_is2re_traj_inf_ood_both/lmln_3_layer_norm_sh/results/2023-08-14-17-23-28-predict_eq2_is2re_traj_inf_ood_both_lmln_3_layer_norm_sh/s2ef_predictions.npz"
    #     ],
    #     ml_path="/private/home/jmu/storage/trajectory_results/oc20data_oc20models_1equiformer/is2re_val_ood/trajs",
    #     dft_path="/private/home/jmu/storage/trajectory_results/oc20data_oc20models_1equiformer/is2re_val_ood/vasp_results/vasp_dirs",
    #     placement_strategy="is2re",
    #     unc_type="max_energy",
    #     err_type="traj_energy",
    #     debug=False,
    #     use_reference=True,
    # )

    with np.load("/private/home/jmu/storage/inference/equiformer_residual_head/val_loss_04/ensemble_head/val_id/results/2023-09-20-09-55-28-residual_ens_eq2_PREDICT/s2ef_predictions.npz") as data:
        data_dict = dict(data)

    unc = []
    err = []

    assert len(unc) == len(err)
    print(f"unc: {len(unc)}, err: {len(err)}")
    print("done")
    