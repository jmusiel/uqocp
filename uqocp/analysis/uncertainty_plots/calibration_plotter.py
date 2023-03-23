# import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot
# from uncertainty_toolbox import metrics_calibration, recalibration

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/jovyan/working/uncertainty-toolbox/uncertainty_toolbox")
import metrics_calibration
import recalibration

import json
import numpy as np


def plotter(name, predicted_y, stdev, true_y):
    p_exp, p_obs = (p.flatten() for p in metrics_calibration.get_proportion_lists_vectorized(
            predicted_y, # y predicted
            stdev, # y predicted standard deviation (uncertainty)
            true_y, # y true value
            prop_type='interval',
            num_bins=100
        )
    )

    # plot
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    ax = axs
    ax.plot(p_exp, p_obs, c='orangered', linewidth=2)
    ax.plot([0,1], [0,1], c='k', linewidth=2)
    ax.fill_between(
        p_exp,
        p_obs,
        p_exp,
        alpha=0.7,
        color='white',
        edgecolor='black',
        hatch = '\\\\\\'
    )
    ax.text(0.8, 0.1, f'Miscalibration\nArea: {metrics_calibration.miscalibration_area_from_proportions(p_exp, p_obs):.2f}', transform=ax.transAxes, ha='center', va='center', fontsize=16, backgroundcolor='white')
    ax.set_xlabel('Expected proportion', fontsize=20)
    ax.set_ylabel('Observed proportion', fontsize=20)
    fig.savefig(name+".png", dpi=300)
    print("generated calibration plot " + str(name))
    # print("--- %s seconds ---" % (time.time() - start_time))

def get_self_recalibrated_data(num, predicted_y, stdev, true_y):
    cal_py = predicted_y[:num]
    cal_st = stdev[:num]
    cal_ty = true_y[:num]

    std_recalibrator = recalibration.get_std_recalibrator(
    cal_py, # y predicted
    cal_st, # y predicted standard deviation (uncertainty)
    cal_ty, # y true value
    )

    stdev_recal = std_recalibrator(stdev)

    return (predicted_y[num:], stdev[num:], true_y[num:]), (predicted_y[num:], stdev_recal[num:], true_y[num:])

def get_recalibrated_data(cal_predicted_y, cal_stdev, cal_true_y, predicted_y, stdev, true_y):

    std_recalibrator = recalibration.get_std_recalibrator(
    cal_predicted_y, # y predicted
    cal_stdev, # y predicted standard deviation (uncertainty)
    cal_true_y, # y true value
    )

    stdev_recal = std_recalibrator(stdev)

    return (predicted_y, stdev, true_y), (predicted_y, stdev_recal, true_y)

def plot_self_recal(name, num, predicted_y, stdev, true_y):
    uncal, recal = get_self_recalibrated_data(num, predicted_y, stdev, true_y)
    plotter("/".join(name.split("/")[:-1] + ["uncal_"+name.split("/")[-1]]), *uncal)
    plotter("/".join(name.split("/")[:-1] + ["recal_"+name.split("/")[-1]]), *recal)

def plot_recal(name, cal_predicted_y, cal_stdev, cal_true_y, predicted_y, stdev, true_y):
    uncal, recal = get_recalibrated_data(cal_predicted_y, cal_stdev, cal_true_y, predicted_y, stdev, true_y)
    plotter("uncal_" + name, *uncal)
    plotter("recal_" + name, *recal)

if __name__ == "__main__":
    json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/ood_errors_traj2traj.json"
    with open(json_path, "r") as jsonfile:
        udict = json.load(jsonfile)

    predicted_y = np.array(udict["dft_de"])
    stdev = np.array([i[-1] for i in udict["inf_e_stdev"]])
    true_y = np.array(udict["is2re_de"])
    plot_self_recal("calplotter/s2e_last", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    stdev = np.array([max(i) for i in udict["inf_e_stdev"]])
    true_y = np.array(udict["is2re_de"])
    plot_self_recal("calplotter/s2e_max", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    stdev = np.array([np.mean(i) for i in udict["inf_e_stdev"]])
    true_y = np.array(udict["is2re_de"])
    plot_self_recal("calplotter/s2e_mean", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    stdev = np.array([i[1] for i in udict["inf_e_stdev"]])
    true_y = np.array(udict["is2re_de"])
    plot_self_recal("calplotter/s2e_firststep", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    stdev = np.array([max([max(j) for j in i]) for i in udict["inf_f_stdev"]])
    true_y = np.array(udict["is2re_de"])
    plot_self_recal("calplotter/s2f_max", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    stdev = np.array([np.mean([max(j) for j in i]) for i in udict["inf_f_stdev"]])
    true_y = np.array(udict["is2re_de"])
    plot_self_recal("calplotter/s2f_mean", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    true_y = np.array(udict["is2re_de"])
    stdev = np.array([max([max(j) for j in i]) for i in udict["ens_f_stdev"]])
    plot_self_recal("calplotter/ens_s2f_max", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    true_y = np.array(udict["is2re_de"])
    stdev = np.array([np.mean([max(j) for j in i]) for i in udict["ens_f_stdev"]])
    plot_self_recal("calplotter/ens_s2f_mean", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    true_y = np.array(udict["is2re_de"])
    stdev = np.array([i for i in udict["is2re_de_stdev"]])
    plot_self_recal("calplotter/is2re_de_stdev", 10, predicted_y, stdev, true_y)

    predicted_y = np.array(udict["dft_de"])
    true_y = np.array(udict["is2re_de"])
    stdev = np.array([i for i in udict["is2re_de_stdev"]]) + np.array([max([max(j) for j in i]) for i in udict["ens_f_stdev"]])
    plot_self_recal("calplotter/linearcombo_is2re_de__ens_s2f_max_stdev", 10, predicted_y, stdev, true_y)