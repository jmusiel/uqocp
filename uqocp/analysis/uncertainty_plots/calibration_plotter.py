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
    plotter("/".join(name.split("/")[:-1] + ["uncal_"+name.split("/")[-1]]), *uncal)
    plotter("/".join(name.split("/")[:-1] + ["recal_"+name.split("/")[-1]]), *recal)

def plot_all_is2re_cal_recal(ooddict, caldict, extra_name):
    for name, method in {
        "s2e_max": lambda udict: np.array([max(i) for i in udict["inf_e_stdev"]]),
        "s2f_max": lambda udict: np.array([max([max(j) for j in i]) for i in udict["inf_f_stdev"]]),
        "ens_e_max": lambda udict: np.array([max(i) for i in udict["ens_e_stdev"]]),
        "ens_s2f_max": lambda udict: np.array([max([max(j) for j in i]) for i in udict["ens_f_stdev"]]),
        "is2re_de_stdev": lambda udict: np.array([i for i in udict["is2re_de_stdev"]]),
    }.items():
        plot_recal(
            name="id_ood/"+name+extra_name,
            cal_predicted_y= np.array(caldict["dft_de"]),
            cal_stdev=method(caldict),
            cal_true_y= np.array(caldict["is2re_de"]),
            predicted_y=np.array(ooddict["dft_de"]),
            stdev=method(ooddict),
            true_y=np.array(ooddict["is2re_de"]),
        )


if __name__ == "__main__":
    json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/ood_errors_traj2traj.json"
    # json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/OC22_oc22_traj2traj.json"
    with open(json_path, "r") as jsonfile:
        ooddict = json.load(jsonfile)
    
    json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/id_errors_traj2traj.json"
    with open(json_path, "r") as jsonfile:
        caldict = json.load(jsonfile)


    plot_all_is2re_cal_recal(ooddict, caldict, "_id_ood")
