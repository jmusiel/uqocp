import numpy as np
from ase.io import Trajectory, read
from pqdm.processes import pqdm
import os
import pickle
from tqdm import tqdm
from uqocp.utils.error import get_force_diff


REF_ENERGIES_PKL_PATH = "/large_experiments/opencatalyst/electrocatalysis/relaxations/mapping/2021_02_07_splits_updated/mappings/final_ref_energies_02_07_2021.pkl"
ADSORB_ML_REF_ENERGIES_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/ref_energies_updated_sysid.pkl"
ADSORB_ML_RANDOM_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/random_metadata_by_sid.pkl"
ADSORB_ML_HEURISTIC_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/heur_metadata_by_intsid.pkl"
ADSORBML_SYSID_SPLIT_PKL_PATH = "/large_experiments/opencatalyst/maaps/mappings/sysid_to_split.pkl"
ADSORB_ML_RELEASE_PKL_PATH = "/large_experiments/opencatalyst/maaps/for_release/mappings/oc20dense_mapping.pkl"


def load_adsorbml_data(
        latent_npz_path,
        ml_trajs_dir,
        vasp_sp_dir,
        placement_strategy,
        split_type=None,
        use_reference=True,
        n_jobs=32,
        debug=False,
        clip_labels=None,
    ):
    """
    latent_npz_path: str
        Path to .npz file containing latent representations and labels.
    vasp_sp_dir: str
        Path to directory containing VASP single-point calculations corresponding to latent representations.
    placement_strategy: str
        Either "random" or "heuristic".
    split_type: list
        List of split types to use. If None, all splits are used. Must list containing up to four str: ["id", "ood_both", "ood_cat", "ood_ads"]
    use_reference: bool
        Whether to use reference energies for adsorbate binding energies.
    n_jobs: int
        Number of jobs to use for loading data.
    """
    print("loading adsorbml data")
    if use_reference:
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
    else:
        raise ValueError(f"invalid placement strategy: {placement_strategy}")
    if split_type is not None:
        with open(ADSORBML_SYSID_SPLIT_PKL_PATH, "rb") as f:
            splits = pickle.load(f)

    unique_sid_order, per_system_data = sort_adsorbml_by_sid_and_fid(latent_npz_path, per_atom_keys=["latents"])
    if split_type is not None:
        unique_sid_order, per_system_data = filter_adsorbml_by_split_type(unique_sid_order, per_system_data, split_type, mapping_dict)
    per_system_latents = per_system_data["latents"]

    # get inputs for labels pqdm
    label_inputs = []
    for sid in tqdm(unique_sid_order, "getting adsorbml label inputs"):
        mapping = mapping_dict[sid]
        if use_reference:
            if placement_strategy == "release":
                ref_e = ref_energies[mapping['system_id']]
            else:
                ref_e = ref_energies[mapping[0]]
        else:
            ref_e = 0
        if placement_strategy == "release":
            dft_mapping_path = os.path.join(vasp_sp_dir, f"{mapping['system_id']}_{mapping['config_id']}.traj")
        else:
            dft_mapping_path = os.path.join(vasp_sp_dir, f"{mapping[0]}_{mapping[1]}.traj")
        ml_traj_path = os.path.join(ml_trajs_dir, f"{sid}.traj")
        label_inputs.append((dft_mapping_path, ml_traj_path, ref_e, "energy"))

    # get vasp sp labels
    if debug:
        label_inputs = label_inputs[:10]
        labels = [_adsorbml_dft_parallel_helper(label_input) for label_input in tqdm(label_inputs, "getting adsorbml labels")]
        per_system_latents = per_system_latents[:10]
    else:
        labels = pqdm(label_inputs, _adsorbml_dft_parallel_helper, n_jobs=n_jobs, desc="getting adsorbml labels")

    if clip_labels is not None:
        labels = np.clip(labels, a_min=clip_labels[0], a_max=clip_labels[1])
    return per_system_latents, labels

def sort_adsorbml_by_sid_and_fid(
        npz_path: str,
        per_atom_keys: list[str]=[],
        per_frame_keys: list[str]=[],
):
    """
    npz_path: str
        Path to .npz file, from ocp main.py inference, containing per atom and per frame data.
    per_atom_keys: list[str]
        List of keys to get from per atom data.
    per_frame_keys: list[str]
        List of keys to get from per frame data.
    """
    # get data and enumerate indices
    per_atom_data = {}
    per_frame_data = {}
    with np.load(npz_path) as npz_data:
        ids = npz_data["ids"]
        frame_chunk_idx = npz_data["chunk_idx"]
        for key in per_atom_keys:
            per_atom_data[key] = npz_data[key]
        for key in per_frame_keys:
            per_frame_data[key] = npz_data[key]
    sids = []
    fids = []
    for id_n in tqdm(ids, "getting adsorbml ids"):
        sid, fid = id_n.split("_")
        sids.append(int(sid))
        fids.append(int(fid))
    sids = np.array(sids)
    fids = np.array(fids)

    # get indices sorted by sid first and fid second get sorted system chunk idx
    sorted_indices = np.lexsort((fids, sids), axis=0)
    sorted_sids = sids[sorted_indices]
    sorted_fids = fids[sorted_indices]
    prev_sid = sorted_sids[0]
    system_chunk_idx = []
    for i, sid in tqdm(enumerate(sorted_sids), "getting adsorbml system chunks", total=len(sorted_sids)):
        if sid != prev_sid:
            system_chunk_idx.append(i)
        prev_sid = sid
    system_chunk_idx = np.array(system_chunk_idx)
    unique_sid_order = list(dict.fromkeys(sorted_sids))

    # sort data
    per_system_data = {}
    for key in per_frame_keys:
        per_system_data[key] = per_frame_data[key][sorted_indices]
        assert len(per_system_data[key]) == len(sorted_sids)
    for key in per_atom_keys:
        per_frame_arr = np.split(per_atom_data[key], frame_chunk_idx)
        per_frame_arr = [per_frame_arr[i] for i in sorted_indices]
        per_system_data[key] = per_frame_arr
        assert len(per_system_data[key]) == len(sorted_sids)
        # per_atom_arr = np.concatenate(per_frame_arr, axis=0)
        frame_chunk_idx = np.cumsum([len(frame) for frame in per_frame_arr])[:-1]

    # get per_system data
    for key in per_system_data.keys():
        per_system_data[key] = np.split(per_system_data[key], system_chunk_idx)
        per_system_data[key] = [np.stack(arr) for arr in per_system_data[key]]
        assert len(per_system_data[key]) == len(unique_sid_order)

    return unique_sid_order, per_system_data

def filter_adsorbml_by_split_type(
        unique_sid_order: list[int],
        per_system_data: dict[list],
        split_type: list[str],
        mapping_dict: dict,
    ):
    """
    unique_sid_order: list[int]
        List of unique system ids.
    per_system_data: dict
        Dictionary of per system data.
    split_type: list[str]
        Split types to filter on, list containing up to four str: ["id", "ood_both", "ood_cat", "ood_ads"].
    mapping_dict: dict
        Dictionary mapping system ids to adsorbml mapping ids (adsorbate_bulk_surface).
    """
    # check split types
    if not type(split_type) == list:
        raise TypeError(f"split type must be list, got {type(split_type)}")
    for key in split_type:
        if key not in ["id", "ood_both", "ood_cat", "ood_ads"]:
            raise ValueError(f"split type {key} not recognized, must be one of: ['id', 'ood_both', 'ood_cat', 'ood_ads']")
    with open(ADSORBML_SYSID_SPLIT_PKL_PATH, "rb") as f:
        splits = pickle.load(f)
    temp_per_system_data = {key:[] for key in per_system_data.keys()}
    temp_unique_sid_order = []
    for i, sid in tqdm(enumerate(unique_sid_order), f"filtering adsorbml on split type: {split_type}", total=len(unique_sid_order)):
        mapping = mapping_dict[sid]
        if type(mapping) == tuple:
            mapping_id = mapping[0]
        elif type(mapping) == dict:
            mapping_id = mapping['system_id']
        else:
            raise TypeError(f"mapping must be tuple (heuristic, random) or dict (release), got {type(mapping)}")
        if splits[mapping_id] in split_type:
            for key, value in per_system_data.items():
                temp_per_system_data[key].append(value[i])
            temp_unique_sid_order.append(sid)
    return temp_unique_sid_order, temp_per_system_data


def _adsorbml_dft_parallel_helper(dft_path_tuple):
    dft_mapping_path, ml_traj_path, ref_e, label_type = dft_path_tuple
    target_atoms = Trajectory(dft_mapping_path)[-1]
    predict_atoms = Trajectory(ml_traj_path)[-1]
    label = _get_label(
        predict_atoms=predict_atoms,
        target_atoms=target_atoms,
        label_type=label_type,
        ref_e=ref_e,
    )
    return label


def load_is2re_data(
        latent_trajs_dir,
        vasp_sp_dir,
        label_type="energy",
        use_reference=True,
        n_jobs=32,
        debug=False,
        clip_labels=None,
        which_frame_latents=None,
        return_sids=False,
        atoms_to_calib_and_test="all",
    ):
    """
    latent_trajs_dir: str
        Path to directory containing is2re inference trajectories storing latent representations.
    vasp_sp_dir: str
        Path to directory containing VASP single-point calculations corresponding to is2re inference trajectories.
    n_jobs: int
        Number of jobs to use for loading data.
    """
    print("loading is2re data")
    if use_reference:
        with open(REF_ENERGIES_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)

    latent_trajs = os.listdir(latent_trajs_dir)
    vasp_sp_dirs = os.listdir(vasp_sp_dir)

    # get intersection in ids
    latent_ids = [int(i.replace(".traj", "")) for i in tqdm(latent_trajs, "getting is2re latent ids")]
    vasp_ids = [int(i) for i in tqdm(vasp_sp_dirs, "getting is2re vasp ids") if os.path.exists(os.path.join(vasp_sp_dir, i, "vasprun.xml"))]
    sids = np.intersect1d(latent_ids, vasp_ids)

    # get pqdm inputs
    input_tuples = []
    for sid in tqdm(sids, "getting is2re label inputs"):
        if use_reference:
            ref_e = ref_energies[f"random{sid}"]
        else:
            ref_e = 0
        input_tuples.append((sid, latent_trajs_dir, vasp_sp_dir, label_type, ref_e, which_frame_latents, atoms_to_calib_and_test))

    if debug:
        input_tuples = input_tuples[:10]
        results = [_is2re_data_parallel_helper(input_tuple) for input_tuple in input_tuples]
    else:
        results = pqdm(input_tuples, _is2re_data_parallel_helper, n_jobs=n_jobs, desc="getting is2re labels")

    # filter out and print errors
    results = [result for result in results if type(result) == tuple]
    errors = [result for result in results if not type(result) == tuple]
    sids = [sid for sid, result in zip(sids, results) if type(result) == tuple]
    print(f"Number of errors: {len(errors)}")
    for i, result in enumerate(errors):
        print(result)
        if i > 10:
            break
    if results == []:
        raise ValueError("No results returned from pqdm")
    
    latents = [result[0] for result in results]
    labels = [result[1] for result in results]

    if clip_labels is not None:
        labels = np.clip(labels, a_min=clip_labels[0], a_max=clip_labels[1])

    if return_sids:
        return latents, labels, sids
    else:
        return latents, labels

def _is2re_data_parallel_helper(is2re_tuple):
    sid, latent_trajs_dir, vasp_sp_dir, label_type, ref_e, which_frame_latents, atoms_to_calib_and_test = is2re_tuple

    latent_traj = Trajectory(os.path.join(latent_trajs_dir, f"{sid}.traj"))
    predict_atoms = latent_traj[-1]
    target_atoms = read(os.path.join(vasp_sp_dir, str(sid), "vasprun.xml"))

    label = _get_label(
        predict_atoms=predict_atoms,
        target_atoms=target_atoms,
        label_type=label_type,
        ref_e=ref_e,
    )

    if which_frame_latents is not None:
        latent = np.array(latent_traj[which_frame_latents].info["latents"])
        if atoms_to_calib_and_test == "all":
            pass
        elif atoms_to_calib_and_test == "both":
            latent = latent[latent_traj[0].get_tags() > 0]
        elif atoms_to_calib_and_test == "adsorbate":
            latent = latent[latent_traj[0].get_tags() == 2]
        elif atoms_to_calib_and_test == "surface":
            latent = latent[latent_traj[0].get_tags() == 1]
        else:
            raise ValueError(f"invalid atoms_to_calib_and_test: {atoms_to_calib_and_test}")
    else:
        latent = np.array([atms.info["latents"] for atms in latent_traj])

    return latent, label

def _get_label(predict_atoms, target_atoms, label_type, ref_e):
    if label_type == "energy":
        target = target_atoms.get_potential_energy() - ref_e
        predict = predict_atoms.get_potential_energy()
        label = np.abs(target - predict)
    elif label_type == "fmax":
        target_forces = target_atoms.get_forces()
        target = np.sqrt((target_forces**2).sum(axis=1).max())
        predict_forces = predict_atoms.get_forces()
        predict = np.sqrt((predict_forces**2).sum(axis=1).max())
        label = target - predict
    elif label_type in ["l2", "l1", "mag", "cos"]:
        target = target_atoms.get_forces()
        target = target[[i for i, a in enumerate(target_atoms) if i not in target_atoms.constraints[0].index]]
        predict = predict_atoms.get_forces()
        predict = predict[[i for i, a in enumerate(predict_atoms) if i not in predict_atoms.constraints[0].index]]
        label = get_force_diff(target, predict, diff_type=label_type)
    else:
        raise ValueError(f"invalid label type: {label_type}")

    return label

def load_npz_helper(filepath):
    with np.load(filepath) as npz:
        latents = npz["latents"]
        energies = npz["energy"]
        chunk_idx = npz["chunk_idx"]
        ids = npz["ids"]
    return latents, energies, chunk_idx, ids

def load_is2re_data_from_err_npz(
    npz_dir,
    err_npz_path,
    use_reference=True,
    n_jobs=32,
    debug=False,
):
    print("loading is2re data")
    if use_reference:
        with open(REF_ENERGIES_PKL_PATH, "rb") as f:
            ref_energies = pickle.load(f)
    
    with np.load(err_npz_path) as npz_data:
        # latents=npz_data["latents"]
        errors=npz_data["errors"]
        sids_e=npz_data["sids"]
        fids_e=npz_data["fids"]
        aids_e=npz_data["aids"]
    err_dict = {sid:err for err, sid in zip(errors, sids_e)}

    filepaths = []
    latents = []
    energies = []
    chunk_idx = []
    ids = []
    sids = []
    fids = []
    aids = []
    per_frame_latents = []
    for filename in tqdm(os.listdir(npz_dir), "loading is2re data from npz_dir"):
        filepath = os.path.join(npz_dir, filename)
        filepaths.append(filepath)
    results = pqdm(filepaths, load_npz_helper, n_jobs=16)
    for lat, en, ch, sfid in results:
        latents.extend(lat)
        energies.extend(en)
        chunk_idx.extend(ch)
        ids.extend(sfid)
        per_frame_latents.extend(np.split(lat, ch))

    # getting sids and fids from ids
    for frame_lat, lat_id in tqdm(zip(per_frame_latents, ids), f"getting sids and fids from ids", total=len(per_frame_latents)):
        sid, fid = lat_id.split("_")
        sids.append(int(sid))
        fids.append(int(fid))
    sids = np.array(sids)
    fids = np.array(fids)

    # get indices sorted by sid first and fid second get sorted system chunk idx
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

    sorted_latents = np.array(per_frame_latents)[sorted_indices]
    persys_latents = np.split(sorted_latents, system_chunk_idx)
    labels = []
    latents = []
    for sid, sys_latent in tqdm(zip(unique_sid_order, persys_latents), "finishing up latents and labels", total=len(unique_sid_order)):
        if sid in err_dict:
            labels.append(err_dict[sid])
            latents.append(np.array(sys_latent.tolist()))

    assert len(labels) == len(latents)
    return latents, labels



if __name__ == "__main__":
    latents, labels = load_is2re_data_from_err_npz(
        npz_dir="/private/home/jmu/storage/inference/inference_on_eq2_relaxed_is2re_trajs/eq2_invariant/val_id/results/2023-08-26-11-31-28-predict_eq2_is2re_traj_inf_eq2_inv_id",
        err_npz_path="/private/home/jmu/storage/inference/inference_on_eq2_relaxed_is2re_trajs/eq2_invariant/val_id/err_npz/errs_latents_frame-1.npz",
        n_jobs=32,
        debug=False,
    )
    print(f"is2re latents: {len(latents)}, labels: {len(labels)}")

    print("done")

