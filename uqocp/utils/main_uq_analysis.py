import argparse
import os
import numpy as np
from tqdm import tqdm
import time

from uqocp.utils.results import (
    get_mean_ensemble_err,
    get_mean_ensemble_unc,
    get_per_traj_ensemble_unc_and_err,
    get_adsorbml_ensemble_uncertainty_random_heuristic_hack,
    get_adsorbml_residual_head_uncertainty_placement_and_id_ood_hack,
    get_meta_npz_results,
    get_adsorbml_ensemble_uncertainty,
    get_unc_err_is2re_residual_head,
)
from uqocp.utils.uncertainty_evaluation import (
    miscalibration_area,
    log_likelihood,
    get_calibration,
    recalibrate,
    auroc,
    spearman,
    error_based_calibration,
    var_z_calibration_test,
    get_sim_nll,
    get_sim_spearman,
    get_sim_nll_and_spearman
)
from uqocp.utils.get_stats import get_runtime, _which_keys
from uqocp.utils.uncertainty_plot import plot_calibration, plot_hexbin, plot_auroc, dump_results, plot_error_based_calibration

"""
Script to evaluate uncertainty quantification methods, takes in uncertainty predictions and measured errors,
and outputs various metrics and plots to evaluate the quality of the uncertainty quantification.
"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vu", 
        "--val_unc_path", 
        nargs="+", 
        help="path to .npz file which stores calibration set uncertainty results",
    )
    parser.add_argument(
        "-ve", 
        "--val_err_path", 
        nargs="+", 
        help="path to .npz file which stores calibration set error results, assumed that cal_unc_path stores both if not given",
    )
    parser.add_argument(
        "-tu", 
        "--test_unc_path", 
        nargs="+", 
        help="path to .npz file which stores test set uncertainty results, if not given then no test set will be used, only calibration set results will be plotted.",
    )
    parser.add_argument(
        "-te", 
        "--test_err_path", 
        nargs="+", 
        help="path to .npz file which stores test set error results, assumed that test_unc_path stores both if not given",
    )
    parser.add_argument(
        "--val_dft_path", 
        help="path to dft trajectories for val set (only for is2re ensemble)",
    )
    parser.add_argument(
        "--test_dft_path", 
        help="path to dft trajectories for test set (only for is2re ensemble)",
    )
    parser.add_argument(
        "-k", 
        "--keys", 
        nargs="+", 
        help="list of checkpoint keys to use for the ensemble, limits what gets used from one large .npz file",
    )
    parser.add_argument(
        "-s", 
        "--savefig_path", 
        help="path to save figures to, defaults to ./figures/[label]",
    )
    parser.add_argument(
        "-l", 
        "--label", 
        help="label to prepend to figure output files",
    )
    parser.add_argument(
        "-d", 
        "--debug", 
        action="store_true", 
        help="run the script quickly on only 10 datapoints to debug it",
    )
    parser.add_argument(
        "-t", 
        "--diff_type", 
        help="type of force error to compute, defaults to 'l2', options are: ['l2' (l2 norm of the difference), 'l1' (l1 norm of the difference), 'mag' (difference in force magnitudes), 'cos' (cosine similarity of the angle between force vectors)]",
    )
    parser.add_argument(
        "-hexp", 
        "--hexbin_percentile", 
        help="Limit for the scope of the hexbin plot limited by a percentile, by default set to 99",
    )
    parser.add_argument(
        "-hexl", 
        "--hexbin_limit", 
        help="Limit for the scope of the hexbin plot limited by an absolute value, by default set to 1",
    )
    parser.add_argument(
        "-rt", 
        "--roc_thresholds",
        nargs="+",
        help="List of thresholds for the receiver operating curve to evaluate at, by default it is set to: 0.05 0.1 0.5",
    )
    parser.add_argument(
        "-u", 
        "--units", 
        help="string to insert for the units, will appear on various plot labels, default is an empty string",
    )
    parser.add_argument(
        "-b", 
        "--betas",
        nargs="+",
        help="if this calibration has been run before, skip the expensive optimization and use given betas, should be a list of floats with exactly two entries",
    )
    parser.add_argument(
        "-uq",
        "--uncertainty_quantity",
        help="type of uncertainty to quantify, options are ['max_l2', 'mean_l2', 'max_energy', 'mean_energy']",
    )
    parser.add_argument(
        "-eq",
        "--error_quantity",
        help="type of error to quantify, options are ['energy', 'l2', 'l1', 'mag', 'cos']",
    )
    parser.add_argument(
        "-m",
        "--uncertainty_method",
        help="method of uncertainty quantification from which to extract uncertainty, options are ['ensemble', 'bayesian', 'meta', latent_distance, 'traj_ensemble], defaults to 'ensemble'",
    )
    parser.add_argument(
        "-cs",
        "--calibration_split",
        help="integer: specify a number of datapoints to use from the validation dataset as the calibration set, and use the remainder as the test set, instead of giving a path to test data, overrides --test_unc_path",
    )
    parser.add_argument(
        "-opt",
        "--optimizer",
        help="optimizer used NLL minimization during recalibration, default: Nelder-Mead"
    )
    parser.add_argument(
        "-dft",
        "--dft_path",
        help="path to dft source data when using adsorbml uncertainty methods, if given then the error will be computed against the dft data",
    )
    parser.add_argument(
        "-f",
        "--split_type_filter",
        help="list of domains to filter the split type by, options to include are ['id', 'ood_both', 'ood_cat', 'ood_ads']. Any included will be used to for the calibration split, and any not included will be used for the test split. Should pass all .npz paths to both --test_unc_path and --val_unc_path",
        nargs="+",
    )
    parser.add_argument(
        "--regularize_cutoff",
        help="eV at which to cutoff for a regularized calibration",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--calculate_adsorption_error",
        type=str,
        default="n",
        help="whether to calculate adsorption error or not (adsorption energy relative to desorbed initial structure), options are (y/n)",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=1000,
        help="number of error sims to run (sim NLL, sim spearman), default 1000",
    )
    parser.add_argument(
        "--simulate_errors",
        type=str,
        default=None,
        help="obsolete, don't use, will print if used",
    )
    parser.add_argument(
        "--save_data",
        type=str,
        default="y",
        help="y or no if you want to save unc, err, and recal_unc to a file (default n)",
    )

    
    return parser

def main(arg_config):
    main_start_time = time.time()

    val_unc_path = ["/private/home/jmu/storage/ensemble_results/trimmed_npz/val_trimmed.npz"]
    if arg_config["val_unc_path"] is not None:
        val_unc_path = arg_config['val_unc_path']
    if len(val_unc_path) == 1:
        val_unc_path = val_unc_path[0]

    val_err_path = val_unc_path
    if arg_config["val_err_path"] is not None:
        val_err_path = arg_config['val_err_path']
    if len(val_err_path) == 1:
        val_err_path = val_err_path[0]
    
    test_unc_path = None
    if arg_config["test_unc_path"] is not None:
        test_unc_path = arg_config['test_unc_path']
    if test_unc_path is not None and len(test_unc_path) == 1:
        test_unc_path = test_unc_path[0]

    test_err_path = test_unc_path
    if arg_config["test_err_path"] is not None:
        test_err_path = arg_config['test_err_path']
    if test_err_path is not None and len(test_err_path) == 1:
        test_err_path = test_err_path[0]

    val_dft_path = None
    if arg_config["val_dft_path"] is not None:
        val_dft_path = arg_config['val_dft_path']
    
    test_dft_path = None
    if arg_config["test_dft_path"] is not None:
        test_dft_path = arg_config['test_dft_path']

    keys = None
    if arg_config["keys"] is not None:
        keys = arg_config['keys']

    diff_type = "l2"
    if arg_config["diff_type"] is not None:
        diff_type = arg_config['diff_type']

    debug = False
    if arg_config["debug"]:
        debug = True

    hexbin_percentile = 99
    if arg_config["hexbin_percentile"] is not None:
        hexbin_percentile = float(arg_config['hexbin_percentile'])

    hexbin_limit = 1
    if arg_config["hexbin_limit"] is not None:
        hexbin_limit = float(arg_config['hexbin_limit'])

    roc_thresholds = [0.05, 0.1, 0.5]
    if arg_config["roc_thresholds"] is not None:
        roc_thresholds = []
        for value in arg_config['roc_thresholds']:
            roc_thresholds.append(float(value))

    units = ""
    if arg_config["units"] is not None:
        units = arg_config['units']

    betas = None
    if arg_config["betas"] is not None:
        if not len(arg_config['betas']) == 2:
            raise ValueError(f"supplied an incorrect number of parameters for betas: {arg_config['betas']}")
        betas = []
        for b in arg_config['betas']:
            betas.append(float(b))

    uncertainty_quantity = None
    if arg_config["uncertainty_quantity"] is not None:
        uncertainty_quantity = arg_config['uncertainty_quantity']

    error_quantity = None
    if arg_config["error_quantity"] is not None:
        error_quantity = arg_config['error_quantity']

    uncertainty_method = "ensemble"
    if arg_config["uncertainty_method"] is not None:
        uncertainty_method = arg_config['uncertainty_method']

    calibration_split = None
    if arg_config["calibration_split"] is not None:
        calibration_split = int(arg_config['calibration_split'])
    if test_unc_path is not None and calibration_split is not None:
        raise ValueError("conflict with having both test_unc_path and calibration_split! only specify one")
    
    regularize_cutoff = None
    if arg_config["regularize_cutoff"] is not None:
        regularize_cutoff = arg_config['regularize_cutoff']

    optimizer = "Nelder-Mead"
    if arg_config["optimizer"] is not None:
        optimizer = arg_config['optimizer']

    dft_path = None
    if arg_config["dft_path"] is not None:
        dft_path = arg_config['dft_path']

    split_type_filter = ["id", "ood_ads"]
    if arg_config["split_type_filter"] is not None:
        split_type_filter = arg_config['split_type_filter']

    calculate_adsorption_error = arg_config['calculate_adsorption_error'] == "y"

    num_simulations = arg_config['num_simulations']

    if not arg_config['simulate_errors'] is None:
        print(f"WARNING: using unused argument 'simulate_errors': {arg_config['simulate_errors']}")
    
    label = ""
    if arg_config["label"] is not None:
        label = arg_config['label']

    autolabel = []
    autolabel.append(uncertainty_method)
    if dft_path is not None:
        autolabel.append("dft")
    if uncertainty_method == "ensemble":
        autolabel.append(diff_type)
    if uncertainty_quantity is not None:
        autolabel.append(f"u-{uncertainty_quantity}")
    if error_quantity is not None:
        autolabel.append(f"e-{error_quantity}")
    if not label == "":
        autolabel.append(label)
    label = "_".join(autolabel)

    savefig_path = "./figures"
    if arg_config["savefig_path"] is not None:
        savefig_path = arg_config['savefig_path']
    savefig_path = os.path.join(savefig_path, label)
    os.makedirs(savefig_path, exist_ok=True)

    if val_dft_path is not None:
        dft_path = val_dft_path

    #### get calibration (val) data for uncertainty and error ####
    val_unc, val_err = get_unc_err(
        unc_path=val_unc_path,
        err_path=val_err_path,
        keys=keys,
        diff_type=diff_type,
        uncertainty_quantity=uncertainty_quantity,
        error_quantity=error_quantity,
        uncertainty_method=uncertainty_method,
        dft_path=dft_path,
        split_type_filter=split_type_filter,
        calculate_adsorption_error=calculate_adsorption_error,
    )

    testing = False
    if calibration_split is not None:
        testing = True
        # get calibration (val) and evaluation (test) split for uncertainty and error
        all_unc = val_unc
        all_err = val_err
        np.random.seed(1)
        choice = np.random.choice(range(len(all_unc)), size=(calibration_split,), replace=False)
        val_ind = np.zeros(len(all_unc), dtype=bool)
        val_ind[choice] = True
        val_unc = all_unc[val_ind]
        val_err = all_err[val_ind]
        test_ind = ~val_ind
        test_unc = all_unc[test_ind]
        test_err = all_err[test_ind]


    # if debugging:
    if debug:
        val_unc = val_unc[:100]
        val_err = val_err[:100]

    # fit calibration
    if betas is None:
        beta_init = [1, 0]
        cal_options= {'maxiter': 200}
        lambdas = [lambda x: x, lambda x: 1]
    else:
        beta_init = betas
        cal_options= {'maxiter': 0}
        lambdas = [lambda x: x, lambda x: 1]

    calibration_results = error_based_calibration(val_unc, val_err)

    if regularize_cutoff is not None:
        reg_val_unc = np.clip(val_unc, 0, regularize_cutoff)
        reg_val_err = np.clip(val_err, 0, regularize_cutoff)
        regularized_calibration_results = error_based_calibration(reg_val_unc, reg_val_err)

    # do test evaluation if test set results are given
    if test_unc_path is not None:
        testing = True

        # invert split type filter, for splitting on domain
        test_split_type_filter = []
        for stype in ["id", "ood_both", "ood_cat", "ood_ads"]:
            if stype not in split_type_filter:
                test_split_type_filter.append(stype)
        split_type_filter = test_split_type_filter

        if test_dft_path is not None:
            dft_path = test_dft_path

        # get evaluation (test) data for uncertainty and error
        test_unc, test_err = get_unc_err(
            unc_path=test_unc_path,
            err_path=test_err_path,
            keys=keys,
            diff_type=diff_type,
            uncertainty_quantity=uncertainty_quantity,
            error_quantity=error_quantity,
            uncertainty_method=uncertainty_method,
            dft_path=dft_path,
            split_type_filter=split_type_filter,
            calculate_adsorption_error=calculate_adsorption_error,
        )

    # do testing first in case it takes a long time
    if testing:
        # if debugging:
        if debug:
            test_unc = test_unc[:100]
            test_err = test_err[:100]

        if regularize_cutoff is not None:
            reg_test_unc = np.clip(test_unc, 0, regularize_cutoff)
            reg_test_err = np.clip(test_err, 0, regularize_cutoff)

        if regularize_cutoff is not None:
            start_time = time.time()
            print(f"----------starting reg test ({time.time()-main_start_time} seconds after starting main)----------")
            log_all(
                unc_path=test_unc_path,
                unc=reg_test_unc,
                err=reg_test_err,
                calibration_results=regularized_calibration_results,
                keys=keys,
                savefig_path=savefig_path,
                label=label+"_reg_test",
                hexbin_percentile=hexbin_percentile,
                hexbin_limit=hexbin_limit,
                roc_thresholds=roc_thresholds,
                units=units,
                num_simulations=num_simulations,
                config=arg_config,
            )
            print(f"----------finished reg test ({time.time()-start_time} seconds)----------")

        start_time = time.time()
        print(f"----------starting test ({time.time()-main_start_time} seconds after starting main)----------")
        log_all(
            unc_path=test_unc_path,
            unc=test_unc,
            err=test_err,
            calibration_results=calibration_results,
            keys=keys,
            savefig_path=savefig_path,
            label=label+"_test",
            hexbin_percentile=hexbin_percentile,
            hexbin_limit=hexbin_limit,
            roc_thresholds=roc_thresholds,
            units=units,
            num_simulations=num_simulations,
            config=arg_config,
        )
        print(f"----------finished test ({time.time()-start_time} seconds)----------")

    if regularize_cutoff is not None:
        start_time = time.time()
        print(f"----------starting reg val ({time.time()-main_start_time} seconds after starting main)----------")
        log_all(
            unc_path=val_unc_path,
            unc=reg_val_unc,
            err=reg_val_err,
            calibration_results=regularized_calibration_results,
            keys=keys,
            savefig_path=savefig_path,
            label=label+"_reg_val",
            hexbin_percentile=hexbin_percentile,
            hexbin_limit=hexbin_limit,
            roc_thresholds=roc_thresholds,
            units=units,
            num_simulations=num_simulations,
            config=arg_config,
        )
        print(f"----------finished reg val ({time.time()-start_time} seconds)----------")

    # now do val
    start_time = time.time()
    print(f"----------starting val ({time.time()-main_start_time} seconds after starting main)----------")
    log_all(
        unc_path=val_unc_path,
        unc=val_unc,
        err=val_err,
        calibration_results=calibration_results,
        keys=keys,
        savefig_path=savefig_path,
        label=label+"_val",
        hexbin_percentile=hexbin_percentile,
        hexbin_limit=hexbin_limit,
        roc_thresholds=roc_thresholds,
        units=units,
        num_simulations=num_simulations,
        config=arg_config,
    )
    print(f"----------finished val ({time.time()-start_time} seconds)----------")


    print(f"done: {label}")

def get_unc_err(
        unc_path,
        err_path,
        keys,
        diff_type,
        uncertainty_quantity,
        error_quantity,
        uncertainty_method,
        dft_path,
        split_type_filter,
        calculate_adsorption_error,
    ):
    if uncertainty_method == "ensemble":
        unc = get_mean_ensemble_unc(
            unc_path, 
            keys=keys, 
            diff_type=diff_type
        )
        err = get_mean_ensemble_err(
            err_path, 
            keys=keys, 
            diff_type=diff_type
        )
    elif uncertainty_method == "traj_ensemble":
        unc, err = get_per_traj_ensemble_unc_and_err(
            traj_dir=unc_path,
            unc_type=uncertainty_quantity,
            err_type=error_quantity,
            dft_path=dft_path,
        )
    elif uncertainty_method == "adsorbml_ensemble":
        unc, err = get_adsorbml_ensemble_uncertainty_random_heuristic_hack(
            unc_path_list=unc_path,
            ml_path=err_path,
            dft_path=dft_path,
            unc_type=uncertainty_quantity,
            err_type=error_quantity,
        )
    elif uncertainty_method == "is2re_ensemble":
        unc, err = get_adsorbml_ensemble_uncertainty(
            unc_path_list=unc_path,
            ml_path=err_path,
            dft_path=dft_path,
            placement_strategy="is2re",
            unc_type=uncertainty_quantity,
            err_type=error_quantity,
            calculate_adsorption_error=calculate_adsorption_error,
        )
    elif uncertainty_method == "adsorbml_residual_head":
        unc, err = get_adsorbml_residual_head_uncertainty_placement_and_id_ood_hack(
            all_unc_path_list=unc_path,
            ml_path=err_path,
            dft_path=dft_path,
            split_type=split_type_filter,
            unc_type=uncertainty_quantity,
            err_type=error_quantity,
        )
    elif uncertainty_method == "is2re_residual_head":
        unc, err = get_unc_err_is2re_residual_head(
            unc_path=unc_path,
            ml_path=err_path,
            dft_path=dft_path,
            unc_type=uncertainty_quantity,
            err_type=error_quantity,
            use_reference=True,
        )
    elif uncertainty_method == "meta" or uncertainty_method == "latent_distance":
        unc, err = get_meta_npz_results(unc_path)
    else:
        raise ValueError(f"invalid uncertainty method: {uncertainty_method}")

    return unc, err

def log_all(
    unc_path,
    unc,
    err,
    calibration_results,
    keys,
    savefig_path,
    label,
    hexbin_percentile,
    hexbin_limit,
    roc_thresholds,
    units,
    num_simulations,
    config,
):
    # recalibrate
    calibration_coefficients = [calibration_results["slope"], calibration_results["intercept"]]
    recal_unc = recalibrate(unc, calibration_coefficients)

    print(f"calibration results: {calibration_results}")
    print(f"calibrating uncertainty: {len(unc)}")

    # save unc and err data to numpy file
    if config["save_data"] == "y":
        filepath = os.path.join(savefig_path, f"{label}_unc_err_recal_results.npz")
        np.savez(filepath, unc=unc, err=err, recal_unc=recal_unc)

    # get uncalibrated uncertainty metrics
    results_mc_uncal = miscalibration_area(unc=unc, err=err)
    results_ll_uncal = log_likelihood(unc=unc, err=err)

    # get recalibrated uncertainty metrics
    results_mc_recal = miscalibration_area(unc=recal_unc, err=err)
    results_ll_recal = log_likelihood(unc=recal_unc, err=err)

    if config['uncertainty_method'] is None or config['uncertainty_method'] == "ensemble":
        # get the full keys corrsponding to this run
        with np.load(unc_path) as data:
            full_keys = [full_key for full_key in data]
        full_keys.remove("dft")
        if keys is not None:
            full_keys = _which_keys(full_keys, keys)

        # get forward pass runtime

        if ".npz" in unc_path:
            try:
                runtime_results = get_runtime(
                    npz_path=unc_path,
                    keys=keys,
                )
            except Exception as e:
                print(f"error getting runtime: {e}")
                runtime_results = None
    else:
        runtime_results = None
        full_keys = None

    # get spearman correlation results
    spearman_results = spearman(
        unc=unc,
        err=err,
    )

    # get receiver operating characteristic results
    auroc_results = [auroc(unc=unc, err=err, err_threshold=i) for i in tqdm(roc_thresholds, "roc results")]

    # get var_z_ci calibration test
    uncal_var_z_ci = var_z_calibration_test(unc=unc, err=err)
    print(f"uncal var_z_ci: {uncal_var_z_ci}")
    recal_var_z_ci = var_z_calibration_test(unc=recal_unc, err=err)
    print(f"recal var_z_ci: {recal_var_z_ci}")

    uncal_sim_nll, uncal_sim_nll_stdev, uncal_sim_spearman, uncal_sim_spearman_stdev = get_sim_nll_and_spearman(unc=unc, num_simulations=num_simulations)
    print(f"uncal sim nll mean: {uncal_sim_nll}, stdev: {uncal_sim_nll_stdev}")
    print(f"uncal sim spearman mean: {uncal_sim_spearman}, stdev: {uncal_sim_spearman_stdev}")
    recal_sim_nll, recal_sim_nll_stdev, recal_sim_spearman, recal_sim_spearman_stdev = get_sim_nll_and_spearman(unc=recal_unc, num_simulations=num_simulations)
    print(f"recal sim nll mean: {recal_sim_nll}, stdev: {recal_sim_nll_stdev}")
    print(f"recal sim spearman mean: {recal_sim_spearman}, stdev: {recal_sim_spearman_stdev}")

    # plot calibration:
    plot_calibration(
        results_mc_uncal=results_mc_uncal,
        results_mc_recal=results_mc_recal,
        results_ll_uncal=results_ll_uncal,
        results_ll_recal=results_ll_recal,
        savefig_path=savefig_path,
        label=label,
        runtime_results=runtime_results,
        spearman_results=spearman_results,
        mae=np.mean(err),
        mue=np.mean(unc).item(),
        cmue=np.mean(recal_unc).item(),
        uncal_var_z_ci=uncal_var_z_ci,
        uncal_sim_nll=uncal_sim_nll,
        uncal_sim_spearman=uncal_sim_spearman,
        recal_var_z_ci=recal_var_z_ci,
        recal_sim_nll=recal_sim_nll,
        recal_sim_spearman=recal_sim_spearman,
    )
    # plot hexbin:
    plot_hexbin(
        unc=unc,
        recal_unc=recal_unc,
        err=err,
        savefig_path=savefig_path,
        label=label+" (99.9999th percentile)",
        units=units,
        percentile=99.9999,
        limit=None,
    )
    plot_hexbin(
        unc=unc,
        recal_unc=recal_unc,
        err=err,
        savefig_path=savefig_path,
        label=f"{label} ({hexbin_percentile}th percentile)",
        units=units,
        percentile=hexbin_percentile,
        limit=None,
    )
    try:
        plot_hexbin(
            unc=unc,
            recal_unc=recal_unc,
            err=err,
            savefig_path=savefig_path,
            label=f"{label} ({hexbin_limit}{units.replace('/','')} limit)",
            units=units,
            percentile=None,
            limit=hexbin_limit,
        )
    except:
        pass
    try:
        plot_hexbin(
            unc=unc,
            recal_unc=recal_unc,
            err=err,
            savefig_path=savefig_path,
            label=f"{label} ({0.2}{units.replace('/','')} limit)",
            units=units,
            percentile=None,
            limit=0.2,
        )
    except: 
        pass

    # plot AU-ROC
    plot_auroc(
        results_auroc=auroc_results,
        savefig_path=savefig_path,
        label=label,
        units=units,
    )

    # plot error based calibration
    uncal_ebc = error_based_calibration(unc, err)
    plot_error_based_calibration(
        ebc_results=uncal_ebc,
        savefig_path=savefig_path,
        label=label+"_uncal",
        units=units,
        limit=None,
    )
    recal_ebc = error_based_calibration(recal_unc, err)
    plot_error_based_calibration(
        ebc_results=recal_ebc,
        savefig_path=savefig_path,
        label=label+"_cal",
        units=units,
        limit=None,
    )

    # dump results
    dump_results(
        results_mc_uncal=results_mc_uncal,
        results_mc_recal=results_mc_recal,
        results_ll_uncal=results_ll_uncal,
        results_ll_recal=results_ll_recal,
        auroc_results=auroc_results,
        spearman_results=spearman_results,
        calibration_results=calibration_results,
        runtime_results=runtime_results,
        models=full_keys,
        savefig_path=savefig_path,
        label=label,
        config=config,
        uncal_ebc=uncal_ebc,
        recal_ebc=recal_ebc,
        uncal_var_z_ci=uncal_var_z_ci,
        uncal_sim_nll=(uncal_sim_nll, uncal_sim_nll_stdev),
        uncal_sim_spearman=(uncal_sim_spearman, uncal_sim_spearman_stdev),
        recal_var_z_ci=recal_var_z_ci,
        recal_sim_nll=(recal_sim_nll, recal_sim_nll_stdev),
        recal_sim_spearman=(recal_sim_spearman, recal_sim_spearman_stdev),
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    arg_config = vars(args)
    main(arg_config)