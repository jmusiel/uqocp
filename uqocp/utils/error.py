import numpy as np
from numpy.linalg import norm
import torch


def get_force_diff(forces0, forces1, diff_type):
    if diff_type == "l2":
        return force_l2_norm_err(forces0, forces1)
    elif diff_type == "l1":
        return force_l1_norm_err(forces0, forces1)
    elif diff_type == "mag":
        return force_magnitude_err(forces0, forces1)
    elif diff_type == "cos":
        return force_cos_sim(forces0, forces1)
    elif diff_type == "energy":
        return energy_err(forces0, forces1)
    else:
        raise ValueError(f"invalid diff type: {diff_type}")

def mean_forces(forces_list):
    return np.mean(forces_list, axis=0)


def force_l2_norm_err(forces0, forces1):
    diff_array = forces0 - forces1
    l2_norm_vector = norm(diff_array, axis=1)
    return l2_norm_vector

def force_l1_norm_err(forces0, forces1):
    diff_array = np.abs(forces0 - forces1)
    l1_norm_vector = np.mean(diff_array, axis=1)
    return l1_norm_vector


def force_magnitude_err(forces0, forces1):
    magnitudes0 = norm(forces0, axis=1)
    magnitudes1 = norm(forces1, axis=1)
    magnitude_err = np.abs(magnitudes0 - magnitudes1)
    return magnitude_err


def force_cos_sim(forces0, forces1):
    # cos_sim = np.dot(forces0, forces1)/(norm(forces0)*norm(forces1)) # cosine similarity formula, but doesn't work elementwise
    dot_array = np.einsum("ij, ij->i", forces0, forces1)
    magnitudes0 = norm(forces0, axis=1)
    magnitudes1 = norm(forces1, axis=1)
    magnitudes = magnitudes0 * magnitudes1
    cos_sim = np.divide(dot_array, magnitudes, out=np.zeros_like(dot_array), where=magnitudes!=0)
    cos_sim = np.abs(cos_sim - 1)
    return cos_sim


def energy_err(energy0, energy1):
    return np.abs(energy0 - energy1)