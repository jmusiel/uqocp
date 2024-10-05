import matplotlib.pyplot as plt
import os
import numpy as np
import json
from sklearn.metrics import r2_score


def plot_calibration(
        results_mc_uncal: dict,
        results_mc_recal: dict,
        results_ll_uncal: dict,
        results_ll_recal: dict,
        savefig_path: str,
        label: str,
        runtime_results: dict=None,
        spearman_results: dict=None,
        mae: float=None,
        mue: float=None,
        cmue: float=None,
        uncal_var_z_ci: tuple=None,
        uncal_sim_nll: float=None,
        uncal_sim_spearman: float=None,
        recal_var_z_ci: tuple=None,
        recal_sim_nll: float=None,
        recal_sim_spearman: float=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot([0,1],[0,1], c="k", label="parity")
    ax.plot(results_mc_uncal["thresholds"], results_mc_uncal["fraction_under_thresholds"], label=f"uncalibrated, miscal={results_mc_uncal['miscalibration_area']:.4f}")
    ax.plot(results_mc_recal["thresholds"], results_mc_recal["fraction_under_thresholds"], label=f"recalibrated, miscal={results_mc_recal['miscalibration_area']:.4f}")
    ax.legend()
    ax.set_xlabel("thresholds")
    ax.set_ylabel("fraction under thresholds")
    ax.set_title(f"{label}")
    text_str = f"before calibration:\n" + \
        f"  miscal area: {results_mc_uncal['miscalibration_area']:.4f}\n" + \
        f"  NLL (mean): {results_ll_uncal['average_log_likelihood']:.4f}\n" + \
        f"after calibration:\n" + \
        f"  miscal area: {results_mc_recal['miscalibration_area']:.4f}\n" + \
        f"  NLL (mean): {results_ll_recal['average_log_likelihood']:.4f}\n" + \
        f"(optimum mean NLL: {results_ll_recal['average_optimal_log_likelihood']:.4f})"
    plt.text(-0.1, 1.2, text_str)
    text_str2 = ""
    if runtime_results:
        text_str2 += f"GPU sec per sample: {runtime_results['total_gpu_seconds_per_sample']:.4f}\n"
        text_str2 += f"samples per GPU sec: {1/runtime_results['total_gpu_seconds_per_sample']:.4f}\n"
    if spearman_results:
        text_str2 += f"Spearman corr.: {spearman_results['spearman_rho']:.4f}\n"
    if mae:
        text_str2 += f"mean err.: {mae:.4f}\n"
    if mue:
        text_str2 += f"mean unc.: {mue:.4f}\n"
    if cmue:
        text_str2 += f"recal mean unc.: {cmue:.4f}\n"
    plt.text(0.35, 1.2, text_str2)
    text_str3 = ""
    if uncal_var_z_ci:
        text_str3 += f"uncal var(Z) ci: [{uncal_var_z_ci[0]:.2f},{uncal_var_z_ci[1]:.2f}]\n"
    if uncal_sim_nll:
        text_str3 += f"uncal sim NLL: {uncal_sim_nll:.4f}\n"
    if uncal_sim_spearman:
        text_str3 += f"uncal sim Spearman: {uncal_sim_spearman:.4f}\n"
    if recal_var_z_ci:
        text_str3 += f"recal var(Z) ci: [{recal_var_z_ci[0]:.2f},{recal_var_z_ci[1]:.2f}]\n"
    if recal_sim_nll:
        text_str3 += f"recal sim NLL: {recal_sim_nll:.4f}\n"
    if recal_sim_spearman:
        text_str3 += f"recal sim Spearman: {recal_sim_spearman:.4f}\n"
    plt.text(0.7, 1.2, text_str3)
    plt.tight_layout()
    fig.savefig(os.path.join(savefig_path, f"{label}_cal_curve.png"), dpi=200)

def plot_hexbin(
    unc: list,
    recal_unc: list,
    err: list,
    savefig_path: str,
    label: str,
    units: str,
    percentile: float=None,
    limit: float=None,
):
    extent = None
    if percentile:
        extent_max = max(np.percentile(unc, percentile), np.percentile(recal_unc, percentile), np.percentile(err,percentile))
        extent = (0, extent_max, 0, extent_max)
    if limit:
        extent = (0, limit, 0, limit)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    hb = axs[0].hexbin(
        unc,
        err,
        gridsize=100,
        mincnt=1,
        extent=extent,
        bins="log",
    )
    axs[0].plot([extent[0],extent[1]],[extent[2],extent[3]], c="k", label="parity")
    axs[0].set_xlabel(f"Predicted Uncertainty {units}")
    axs[0].set_ylabel(f"Measured Error {units}")
    axs[0].set_title(f"uncalibrated")
    hb = axs[1].hexbin(
        recal_unc,
        err,
        gridsize=100,
        mincnt=1,
        extent=extent,
        bins="log",
    )
    axs[1].plot([extent[0],extent[1]],[extent[2],extent[3]], c="k", label="parity")
    axs[1].set_xlabel(f"Predicted Uncertainty {units}")
    axs[0].set_ylabel(f"Measured Error {units}")
    axs[1].set_title(f"recalibrated")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(hb, cax=cbar_ax)
    fig.suptitle(f"{label}")
    fig.savefig(os.path.join(savefig_path, f"{label}_hexbin_parity.png"), dpi=200)

def plot_auroc(
    results_auroc: list[dict],
    savefig_path: str,
    label: str,
    units: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for result in results_auroc:
        ax.plot(result["fpr"], result["tpr"], label=f"err thresh: {result['err_threshold']} {units}, AUC={round(result['auc'],4)}")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    ax.set_title(f"{label} ROC curve")
    ax.legend()
    fig.savefig(os.path.join(savefig_path, f"{label}_roc_curve.png"), dpi=200)

def dump_results(
    results_mc_uncal: dict,
    results_mc_recal: dict,
    results_ll_uncal: dict,
    results_ll_recal: dict,
    auroc_results: list[dict],
    spearman_results: dict,
    calibration_results: dict,
    runtime_results: dict,
    models: list[str],
    savefig_path: str,
    label: str,
    config: dict,
    uncal_ebc: dict,
    recal_ebc: dict,
    uncal_var_z_ci: tuple,
    uncal_sim_nll: tuple,
    uncal_sim_spearman: tuple,
    recal_var_z_ci: tuple,
    recal_sim_nll: tuple,
    recal_sim_spearman: tuple,
):
    results_mc_uncal["thresholds"] = results_mc_uncal["thresholds"].tolist()
    results_mc_recal["thresholds"] = results_mc_recal["thresholds"].tolist()
    calibration_results_dict = {}
    for key, value in calibration_results.items():
        if key == "x":
            value = value.tolist()
        if key == "final_simplex":
            value = [x.tolist() for x in value]
        if key == "jac":
            value = value.tolist()
        if key == "hess_inv":
            value = value.tolist()
        calibration_results_dict[key] = value

    sim_results = {
        "uncal":{
            "var_z_ci": uncal_var_z_ci,
            "sim_nll": {
                "mean": uncal_sim_nll[0],
                "stdev": uncal_sim_nll[1],
            },
            "sim_spearman": {
                "mean": uncal_sim_spearman[0],
                "stdev": uncal_sim_spearman[1],
            },
        },
        "recal":{
            "var_z_ci": recal_var_z_ci,
            "sim_nll": {
                "mean": recal_sim_nll[0],
                "stdev": recal_sim_nll[1],
            },
            "sim_spearman": {
                "mean": recal_sim_spearman[0],
                "stdev": recal_sim_spearman[1],
            },
        },
    }

    results_dict = {
        "config": config,
        "mc_uncal": results_mc_uncal,
        "ll_uncal": results_ll_uncal,
        "mc_recal": results_mc_recal,
        "ll_recal": results_ll_recal,
        "auroc_results": auroc_results,
        "spearman_results": spearman_results,
        "calibration_results": calibration_results_dict,
        "runtime_results": runtime_results,
        "models": models,
        "uncal_ebc": uncal_ebc,
        "recal_ebc": recal_ebc,
        "sim_results": sim_results,
    }
                    
    filepath = os.path.join(savefig_path, f"{label}_results.json")
    results_dict = make_json_serializable(results_dict)
    with open(filepath, "w") as f:
        json.dump(results_dict, f)

def make_json_serializable(some_dict, keys=[]):
    # recursively convert values in nested dictionary to serializable types
    if type(some_dict) == dict:
        for key, value in some_dict.items():
            try:
                json.dumps(value)
            except TypeError:
                some_dict[key] = make_json_serializable(value, [k for k in keys] + [key])
    else:
        if "float" in some_dict.__class__.__name__:
            some_dict = float(some_dict)
        elif "ndarray" in some_dict.__class__.__name__:
            some_dict = some_dict.tolist()
        else:
            raise TypeError(f"key {'.'.join(keys)} has value {some_dict} of type {type(some_dict)} which is not serializable")
    return some_dict

def plot_error_based_calibration(
    ebc_results,
    savefig_path: str,
    label: str,
    units: str,
    limit: float=None,
):

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.set_xlabel(f"RMV ({units})")
    ax.set_ylabel(f"RMSE ({units})")

    extent = None
    if limit:
        extent = (0, limit, 0, limit)

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    min_x = min(ebc_results["rmv"])
    max_x = max(ebc_results["rmv"])
    r2 = r2_score(ebc_results["rmse"], ebc_results["rmv"])
    ax.plot(
        [min_x, max_x], 
        [min_x, max_x], 
        color="black", 
        linestyle="--",
        label=f"Parity: $R^2$ = {r2:.3f}",
    )

    slope = ebc_results["slope"]
    intercept = ebc_results["intercept"]
    fitted_r2 = ebc_results["r2"]
    ax.plot(
        [min_x, max_x],
        [min_x*slope + intercept, max_x*slope + intercept],
        color="red",
        linestyle="--",
        label=f"Fitted: $R^2$ = {fitted_r2:.3f}, slope = {slope:.3f}, intercept = {intercept:.3f}",
    )

    ax.scatter(
        ebc_results["rmv"], 
        ebc_results["rmse"], 
        color="tab:blue", 
    )

    rmse_lower_err = ebc_results["rmse"] - np.array(ebc_results["rmse_ci_low"])
    rmse_higher_err = np.array(ebc_results["rmse_ci_high"]) - ebc_results["rmse"]
    ax.errorbar(
        ebc_results["rmv"].T[0],
        ebc_results["rmse"],
        yerr=(rmse_lower_err, rmse_higher_err),
        fmt='none',
        ecolor='tab:blue',
        capsize=2,
    )

    ax.legend()
    fig.savefig(os.path.join(savefig_path, f"{label}_error_based_calibration.png"), dpi=200)