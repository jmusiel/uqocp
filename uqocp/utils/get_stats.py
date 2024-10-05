import json
import numpy as np
from glob import glob
import os
import yaml


def _which_keys(full_key_strings, given_keys_substrings):
    """
    check if any of the given_keys_substrings appear in full_key_strings, return the keys in full
    :param full_key_strings: The list of keys (full strings) from the .npz file.
    :param given_keys_substrings: The list of substrings given as the 'keys' parameter to be checked against.
    """
    result = []
    for substring in given_keys_substrings:
        for full_string in full_key_strings:
            if substring in full_string:
                result.append(full_string)
    return result

def get_runtime(npz_path, keys=None, verbose=False):
    with np.load(npz_path) as data:
        full_keys = [full_key for full_key in data]
    full_keys.remove("dft")
    if keys is not None:
        full_keys = _which_keys(full_keys, keys)
    
    total_gpu_seconds_per_sample = 0
    total_gpu_seconds_per_batch = 0
    time_dict = {}
    for full_key in full_keys:
        base_path = npz_path.split("ensemble_results")[0]
        paths = glob(os.path.join(base_path, "inference/**/logs/wandb", full_key, "wandb/latest-run/files/wandb-summary.json"), recursive=True)
        if len(paths) == 1:
            path = paths[0]
        else:
            raise ValueError(f"too many paths: {paths}")

        with open(path, "r") as f:
            json_dict = json.load(f)
        runtime = json_dict["_wandb"]["runtime"]

        with open(path.replace("wandb-summary.json", "config.yaml"), "r") as f:
            yaml_dict = yaml.safe_load(f)
        gpus = yaml_dict["slurm"]["value"]["gpus_per_node"] * yaml_dict["slurm"]["value"]["nodes"]

        inference_path = os.path.join(path.split("logs/wandb")[0], "results", full_key, "s2ef_predictions.npz")
        with np.load(inference_path) as data:
            num_samples = len(data["ids"])

        gpu_time = runtime * gpus
        batch_size = yaml_dict["optim"]["value"]["eval_batch_size"]
        num_batches = num_samples/batch_size
        gpu_seconds_per_sample = gpu_time/num_samples
        gpu_seconds_per_batch = gpu_time/num_batches
        total_gpu_seconds_per_sample += gpu_seconds_per_sample
        total_gpu_seconds_per_batch += gpu_seconds_per_batch

        time_dict[full_key] = {
            "runtime": runtime,
            "gpus": gpus,
            "gpu_time": gpu_time,
            "num_samples": num_samples,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "gpu_seconds_per_sample": gpu_seconds_per_sample,
            "gpu_seconds_per_batch": gpu_seconds_per_batch,
        }

    if verbose:
        for key, value in time_dict.items():
            print(f"{key}:")
            for k, v in value.items():
                print(f"\t{k}: {v}")
        print(f"total gpu seconds per sample: {total_gpu_seconds_per_sample}")
        print(f"total gpu seconds per batch: {total_gpu_seconds_per_batch}")

    return {
        "per_model": time_dict,
        "total_gpu_seconds_per_sample": total_gpu_seconds_per_sample, 
        "total_gpu_seconds_per_batch": total_gpu_seconds_per_batch,
    }

    return total_gpu_seconds_per_sample

if __name__ == "__main__":
    val_unc_path = "/private/home/jmu/storage/ensemble_results/all_public_ensemble/val_ood_both_all_public.npz"
    keys = ["m-escn_l6_m3_lay20_all_md_s2ef_d", "m-escn_l6_m2_lay12_all_md_s2ef_d", "m-gnoc_large_s2ef_all_md_d", "m-scn_t4_b2_s2ef_2M_d"]
    results = get_runtime(val_unc_path, keys, verbose=True)
