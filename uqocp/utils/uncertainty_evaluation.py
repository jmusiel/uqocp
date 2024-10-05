

import numpy as np
from scipy import stats
from tqdm import tqdm
from typing import Any, Callable, Dict, List
from scipy.optimize import minimize
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import traceback
import logging
from pqdm.processes import pqdm


def miscalibration_area(unc, err):
    """
    Computes the miscalibration area given the provided uncertainty estimates
    and true errors.
    """
    standard_devs = err/unc

    probabilities = 2 * (stats.norm.cdf(standard_devs) - 0.5)
    sorted_probabilities = sorted(probabilities)

    fraction_under_thresholds = []
    threshold = 0

    for i in tqdm(range(len(sorted_probabilities)), "miscal area"):
        while sorted_probabilities[i] > threshold:
            fraction_under_thresholds.append(i/len(sorted_probabilities))
            threshold += 0.001

    # Condition used 1.0001 to catch floating point errors.
    while threshold < 1.0001:
        fraction_under_thresholds.append(1)
        threshold += 0.001

    thresholds = np.linspace(0, 1, num=1001)
    miscalibration = [np.abs(fraction_under_thresholds[i] - thresholds[i]) for i in range(len(thresholds))]
    miscalibration_area = 0
    for i in range(1, 1001):
        miscalibration_area += np.average([miscalibration[i-1], miscalibration[i]]) * 0.001

    result = {
        'fraction_under_thresholds': fraction_under_thresholds,
        'thresholds': thresholds,
        'miscalibration_area': miscalibration_area
    }
    return result


def log_likelihood(unc, err):
    """
    Computes the log likelihood, average log likelihood, optimal log
    likelihood, and average optimal log likelihood, to produce the
    observed true errors given the provided uncertainty estimates.
    """
    unc = np.array(unc, dtype="float64")
    err = np.array(err, dtype="float64")
    unc = np.clip(unc, 0.001, None)
    log_likelihood_arr = (np.log(2 * np.pi * (unc**2)) / 2) + (err**2/(2 * unc**2))
    optimal_log_likelihood_arr = (np.log(2 * np.pi * (err**2)) / 2) + (1/2)

    log_likelihood = np.sum(log_likelihood_arr)
    optimal_log_likelihood = np.sum(optimal_log_likelihood_arr)

    return {'log_likelihood': log_likelihood,
            'optimal_log_likelihood': optimal_log_likelihood,
            'average_log_likelihood': log_likelihood / len(err),
            'average_optimal_log_likelihood': optimal_log_likelihood / len(err)}

def spearman(unc, err):
    """
    Computes spearman rank correlation coefficient with scipy

    returns:
        rho: spearman rank correlation coefficient (higher is better)
        p: p-value, probability that this result could have been obtained by chance (lower is better)
    """
    rho, p = stats.spearmanr(unc, err)
    return {
        "spearman_rho": rho,
        "spearman_p-value": p,
    }

def auroc(unc, err, err_threshold=1):
    """
    Computes the area under the receiver operating characteristic curve.

    returns:
        fpr: false positive rate (array)
        tpr: true positive rate (array)
        unc_thresholds: corresponding uncertainty thresholds (array)
        err_threshold: given error threshold float
        auc: area under the roc curve (higher is better)
    """
    unc = np.array(unc)
    err = np.array(err)

    high_err_indices = err >= err_threshold
    low_err_indices = err < err_threshold
    err_binary = np.zeros_like(err)
    err_binary[high_err_indices] = 1
    err_binary[low_err_indices] = -1

    fpr, tpr, unc_thresholds = metrics.roc_curve(err_binary, unc)
    auc = metrics.roc_auc_score(err_binary, unc)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "unc_thresholds": unc_thresholds.tolist(),
        "err_threshold": err_threshold,
        "auc": auc,
    }


def get_calibration(
    val_unc: List[float],
    val_err: List[float],
    lambdas: List[Callable]=[lambda x: x, lambda x: 1],
    beta_init: List[float]= [1, 0],
    method="BFGS",
    options={
        'maxiter': 200,
    }
    ):
    """
    Given the stored predictions of an uncertainty estimator and fixed
    transformations, performs a calibration by selecting an optimal
    weighting of the transformations to minimize the objective function.

    For example, to perform a linear calibration:
        [lambda x: x, lambda x: 1]
        With initial weights of [1, 0]

    parameters:
        val_err: list of errors for calibrating on.
        val_unc: list of predicted uncertainties for calibrating on.
        lambdas: A list of fixed transformations to apply to the uncalibrated uncertainty estimates.
        beta_init: A list that defines the initial weighting of the transformations, before optimization.
        method: The optimization method to use.
        options: A dictionary of options to pass to the optimization method.
    """

    # Calibrate based on sampled data.
    uncertainty = np.array(val_unc)
    errors = np.array(val_err)

    result = minimize(
        objective_function,
        beta_init,
        args=(uncertainty, errors, lambdas),
        method=method,
        options=options,
    )

    # calibration_coefficients = result.x

    return result

def objective_function(
        beta: List[float],
        uncertainty: List[float],
        errors: List[float],
        lambdas: List[Callable]
    ) -> float:
    """
    Defines the cost imposed (NLL) by a particular calibration.

    parameters:
        beta: The transformation weights used in calibration.
        uncertainty: A list of uncalibrated uncertainty estimates.
        errors: The list of true prediction erros.
        lambdas: The list of transformations used in calibration.
    """
    # Construct prediction through lambdas and betas.
    pred_vars = np.zeros(len(uncertainty))
    beta = np.array(beta, dtype="float64")
    uncertainty = np.array(uncertainty, dtype="float64")
    errors = np.array(errors, dtype="float64")

    for i in range(len(beta)):
        pred_vars += beta[i] * lambdas[i](uncertainty)
    pred_vars = np.clip(pred_vars, 0.00001, None)
    # costs = (np.log(pred_vars) / 2) + (errors**2 / (2 * pred_vars))
    costs = log_likelihood(pred_vars, errors)['average_log_likelihood']
    # costs = miscalibration_area(pred_vars, errors)['miscalibration_area']

    cost = np.sum(costs)
    print(f"costs: {cost}, ll: {log_likelihood(recalibrate(uncertainty, beta, lambdas=lambdas, disable_tqdm=True), errors)['average_log_likelihood']}, betas: {beta}, shapes: {costs.shape}], {pred_vars.shape}")
    return(cost)

def recalibrate(
    given_unc: List[float],
    beta: List[float],
    lambdas: List[Callable]=[lambda x: x, lambda x: 1],
    disable_tqdm = False,
    ) -> List[Dict[str, float]]:
    """
    Calibrates a collection of uncertainty estimates.

    parameters:
        given_unc: given uncertainties.
        beta: Optimized transformation weights.
        lambdas: The list of transformations used in calibration.
        disable_tqdm: Whether to disable the progress bar.
    """
    calibrated_unc = np.zeros_like(given_unc)
    beta = np.array(beta, dtype="float64")

    for i in tqdm(range(len(beta)), "recalibrating terms", disable=disable_tqdm):
        calibrated_unc += beta[i] * lambdas[i](given_unc)
    return calibrated_unc

def error_based_calibration(
    val_unc: List[float],
    val_err: List[float],
    num_bins: int = 20,
    ):
    """
    Uses the expected correlation between RMSE and RMV to generate a calibration fit, and error based calibration plots.
    An ideal UQ technique should have a slope of 1 and an intercept of 0, and a perfect correlation between RMSE and RMV (R^2 = 1).
    Bin each energy prediction based on predicted uncertainty, and compute the mean error and mean uncertainty for each bin.
    Fit a line through the bins, and use the slope and intercept to calibrate the uncertainty estimates.
    The uncertainty estimates will now have correct calibration (slope = 1, intercept = 0), but the correlation between RMSE and RMV will show how good the uncertainty metric is.
    """

    uncertainty = np.array(val_unc)
    errors = np.array(val_err)

    # sort arrays based on uncertainty
    sorted_indices = np.argsort(uncertainty)
    sorted_uncertainty = uncertainty[sorted_indices]
    sorted_errors = errors[sorted_indices] 

    bin_indices = np.array_split(sorted_indices, num_bins)
    binned_uncertainty = [uncertainty[bin_index_arr] for bin_index_arr in bin_indices]
    binned_errors = [errors[bin_index_arr] for bin_index_arr in bin_indices]

    def calc_rmse(x_bin):
        return np.sqrt(np.mean(x_bin**2))
    
    # compute RMSE and RMV for each bin
    rmv = np.array([calc_rmse(u_bin) for u_bin in binned_uncertainty]).reshape((-1, 1))
    rmse = np.array([calc_rmse(e_bin) for e_bin in binned_errors])
    rmse_ci = [stats.bootstrap(data=(e_bin,), statistic=calc_rmse, vectorized=False) for e_bin in binned_errors]
    rmse_ci_high = np.array([i.confidence_interval.high for i in rmse_ci])
    rmse_ci_low = np.array([i.confidence_interval.low for i in rmse_ci])

    # fit linear regression to RMSE vs RMV
    model = LinearRegression().fit(rmv, rmse)
    slope = model.coef_[0]
    intercept = model.intercept_
    rmse_pred = model.predict(rmv)
    unfit_r2 = r2_score(rmse, rmv)
    r2 = r2_score(rmse, rmse_pred)

    result = {
        "rmv": rmv,
        "rmse": rmse,
        "rmse_ci_high": rmse_ci_high,
        "rmse_ci_low": rmse_ci_low,
        "slope": slope,
        "intercept": intercept,
        "unfit_r2": unfit_r2,
        "r2": r2,
    }
    
    return result

def simulate_errs(unc):
    """
        Get simulated errors assuming normal distribution with stdev equal to the uncertainty
    """
    sim_errs = []
    for stdev in unc:
        error = np.abs(np.random.normal(0, stdev))
        sim_errs.append(error)
    return sim_errs

def parallel_sim_nll_and_spearman(unc):
    sim_errs = simulate_errs(unc)
    sim_nll = log_likelihood(unc, sim_errs)['average_log_likelihood']
    sim_spearman = spearman(unc, sim_errs)['spearman_rho']
    return sim_nll, sim_spearman

def yield_unc(unc, num_simulations):
    for i in range(num_simulations):
        yield unc

def get_sim_nll_and_spearman(unc, num_simulations=1000):
    unc = np.array(unc)
    below_0 = unc[unc < 0]
    at_0 = unc[unc == 0]
    print(f"trying to simulate nll, found {len(below_0)} uncertainties below 0, and {len(at_0)} at 0, out of {len(unc)} total")
    print(f"examples of some below 0: {below_0[:10]}")
    unc = np.clip(np.abs(unc), 0.000001, None)

    results = pqdm(yield_unc(unc, num_simulations), parallel_sim_nll_and_spearman, n_jobs=4, total=num_simulations)

    sim_nlls = []
    sim_spearmans = []
    for sim_nll, sim_spearman in results:
        sim_nlls.append(sim_nll)
        sim_spearmans.append(sim_spearman)

    return np.mean(sim_nlls), np.std(sim_nlls), np.mean(sim_spearmans), np.std(sim_spearmans)

def get_sim_nll(unc, num_simulations=1000):
    unc = np.array(unc)
    below_0 = unc[unc < 0]
    at_0 = unc[unc == 0]
    print(f"trying to simulate nll, found {len(below_0)} uncertainties below 0, and {len(at_0)} at 0, out of {len(unc)} total")
    print(f"examples of some below 0: {below_0[:10]}")
    unc = np.clip(np.abs(unc), 0.0001, None)

    sim_nlls = []
    for i in range(num_simulations):
        sim_errs = simulate_errs(unc)
        sim_nll = log_likelihood(unc, sim_errs)['average_log_likelihood']
        sim_nlls.append(sim_nll)
    return np.mean(sim_nlls), np.std(sim_nlls)

def get_sim_spearman(unc, num_simulations=1000):
    sim_spearmans = []
    for i in range(num_simulations):
        sim_errs = simulate_errs(unc)
        sim_spearmans.append(spearman(unc, sim_errs)['spearman_rho'])
    return np.mean(sim_spearmans), np.std(sim_spearmans)

def var_z_calibration_test(unc, err):
    """
    Computes the variance of the z-score of the errors, which should be 1 if the uncertainties are well-calibrated.
    """
    # to prevent memory issues, set batch size if len(unc) > 50000

    z_scores = err/unc
    def calc_var_z(x):
        return np.var(x)
    
    n_resamples = 1000
    batch_size = 1000
    finished = False
    while not finished:
        try:
            var_z_ci = stats.bootstrap(data=(z_scores,), statistic=calc_var_z, vectorized=False, n_resamples=n_resamples, batch=batch_size)
            finished = True
        except Exception as e:
            logging.error(f"Error in var_z_calibration_test: {e}")
            logging.error(traceback.format_exc())
            if batch_size < 2:
                raise e
            batch_size = batch_size // 2
            logging.error(f"Trying again with batch size {batch_size}")

    var_z_ci = stats.bootstrap(data=(z_scores,), statistic=calc_var_z, vectorized=False, batch=batch_size)
    return var_z_ci.confidence_interval.low, var_z_ci.confidence_interval.high

