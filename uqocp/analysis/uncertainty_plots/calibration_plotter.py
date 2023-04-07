# import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot
# from uncertainty_toolbox import metrics_calibration, recalibration

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/jovyan/working/uncertainty-toolbox/uncertainty_toolbox")
import metrics_calibration
import recalibration

import json
import numpy as np
import os
from tqdm import tqdm

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
            name=extra_name+"/"+name,
            cal_predicted_y= np.array(caldict["dft_de"])+ 1e-8,
            cal_stdev=method(caldict)+ 1e-8,
            cal_true_y= np.array(caldict["is2re_de"])+ 1e-8,
            predicted_y=np.array(ooddict["dft_de"])+ 1e-8,
            stdev=method(ooddict)+ 1e-8,
            true_y=np.array(ooddict["is2re_de"])+ 1e-8,
        )

def plot_all_ens_de_mean_cal_recal(ooddict, caldict, extra_name):
    for name, (key, method) in {
        "s2e_max": ("inf_e_stdev", lambda sysx: max(sysx)),
        "s2f_max": ("inf_f_stdev", lambda sysx: max([max(n) for n in sysx])),
        "ens_e_max": ("ens_e_stdev", lambda sysx: max(sysx)),
        "ens_s2f_max": ("ens_f_stdev", lambda sysx: max([max(n) for n in sysx])),
        "is2re_de_stdev": ("is2re_de_stdev", lambda sysx: sysx),
    }.items():
        kwargs = {}
        keywords = [
            "cal_predicted_y",
            "cal_stdev",
            "cal_true_y",
            "predicted_y",
            "stdev",
            "true_y",
        ]
        for k in keywords:
            kwargs[k] = []
        for i, system_x_cal in tqdm(enumerate(caldict[key]), name):
            if not type(system_x_cal) == float and not type(system_x_cal[0]) == float and len(system_x_cal[0]) == 0:
                print("\nbad index: " + str(i))
            else:
                cal_value = method(system_x_cal)

                kwargs["cal_predicted_y"].append(caldict["dft_de"][i])
                kwargs["cal_stdev"].append(cal_value)
                kwargs["cal_true_y"].append(caldict["ens_de_mean"][i])
    
        for i, system_x_ood in tqdm(enumerate(ooddict[key]), name):
            if not type(system_x_ood) == float and not type(system_x_ood[0]) == float and len(system_x_ood[0]) == 0:
                print("bad index: " + str(i))
            else:
                ood_value = method(system_x_ood)

                kwargs["predicted_y"].append(ooddict["dft_de"][i])
                kwargs["stdev"].append(ood_value)
                kwargs["true_y"].append(ooddict["ens_de_mean"][i])

        print("lengths:")
        print("cal_predicted_y: "+str(len(kwargs["cal_predicted_y"])))
        print("cal_stdev: "+str(len(kwargs["cal_stdev"])))
        print("cal_true_y: "+str(len(kwargs["cal_true_y"])))
        print("predicted_y: "+str(len(kwargs["predicted_y"])))
        print("stdev: "+str(len(kwargs["stdev"])))
        print("true_y: "+str(len(kwargs["true_y"])))

        plot_recal(
            name=extra_name+"/"+name,
            cal_predicted_y=np.array(kwargs["cal_predicted_y"])+ 1e-8,
            cal_stdev=np.array(kwargs["cal_stdev"])+ 1e-8,
            cal_true_y=np.abs(np.array(kwargs["cal_true_y"]))+ 1e-8,
            predicted_y=np.array(kwargs["predicted_y"])+ 1e-8,
            stdev=np.array(kwargs["stdev"])+ 1e-8,
            true_y=np.abs(np.array(kwargs["true_y"]))+ 1e-8,
        )

def flatten(l):
    return [y for x in l for y in x]

def plot_all_s2e_vs_s2f_cal_recal(ooddict, caldict, extra_name):

    cal_predicted_y = np.array(flatten(caldict["inf_e_error"]))+ 1e-8
    predicted_y = np.array(flatten(ooddict["inf_e_error"]))+ 1e-8
    plot_recal(
        name=extra_name+"/"+"s2e_predict_s2e",
        cal_predicted_y= cal_predicted_y,
        cal_stdev= np.array(flatten(caldict["inf_e_stdev"]))+ 1e-8,
        cal_true_y= np.zeros_like(cal_predicted_y),
        predicted_y=predicted_y,
        stdev= np.array(flatten(ooddict["inf_e_stdev"]))+ 1e-8,
        true_y=np.zeros_like(predicted_y),
    )

    plot_recal(
        name=extra_name+"/"+"s2fl2max_predict_s2e",
        cal_predicted_y= cal_predicted_y,
        cal_stdev= np.array(flatten([[max(j) for j in i] for i in caldict["inf_f_stdev"]]))+ 1e-8,
        cal_true_y= np.zeros_like(cal_predicted_y),
        predicted_y=predicted_y,
        stdev= np.array(flatten([[max(j) for j in i] for i in ooddict["inf_f_stdev"]]))+ 1e-8,
        true_y=np.zeros_like(predicted_y),
    )

    cal_predicted_y = np.array(flatten([[max(j) for j in i] for i in caldict["inf_f_l2error"]])) + 1e-8
    predicted_y = np.array(flatten([[max(j) for j in i] for i in ooddict["inf_f_l2error"]]))+ 1e-8
    plot_recal(
        name=extra_name+"/"+"s2e_predict_s2fl2max",
        cal_predicted_y= cal_predicted_y,
        cal_stdev= np.array(flatten(caldict["inf_e_stdev"]))+ 1e-8,
        cal_true_y= np.zeros_like(cal_predicted_y),
        predicted_y=predicted_y,
        stdev= np.array(flatten(ooddict["inf_e_stdev"]))+ 1e-8,
        true_y=np.zeros_like(predicted_y),
    )

    plot_recal(
        name=extra_name+"/"+"s2l2fmax_predict_s2fl2max",
        cal_predicted_y= cal_predicted_y,
        cal_stdev= np.array(flatten([[max(j) for j in i] for i in caldict["inf_f_stdev"]]))+ 1e-8,
        cal_true_y= np.zeros_like(cal_predicted_y),
        predicted_y=predicted_y,
        stdev= np.array(flatten([[max(j) for j in i] for i in ooddict["inf_f_stdev"]]))+ 1e-8,
        true_y=np.zeros_like(predicted_y),
    )



if __name__ == "__main__":
    # json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/ood_errors_traj2traj.json"
    # json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/OC22_oc22_traj2traj.json"
    # json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/Sudheesh_mof_zeolite_errors_traj2traj.json"
    # with open(json_path, "r") as jsonfile:
    #     ooddict = json.load(jsonfile)
    
    json_path = "/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/id_9000_traj2traj.json"
    with open(json_path, "r") as jsonfile:
        caldict = json.load(jsonfile)


    # plot_all_is2re_cal_recal(ooddict, caldict, "id_ood_is2re")
    # plot_all_s2e_vs_s2f_cal_recal(ooddict, caldict, "ood_s2f_vs_s2e")
    # plot_all_s2e_vs_s2f_cal_recal(ooddict, caldict, "szeolite_s2f_vs_s2e")
    # plot_all_is2re_cal_recal(ooddict, caldict, "szeolite_is2re")


    for i, j in [
        ("done9000/ens_de_mean_ood","/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/ood_9000_traj2traj.json"),
        ("done9000/ens_de_mean_oc22","/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/OC22_oc22_traj2traj.json"),
        ("done9000/ens_de_mean_szeolite","/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/Sudheesh_mof_zeolite_errors_traj2traj.json"),
    ]: 
        os.makedirs(i, exist_ok=True)
        json_path = j
        with open(json_path, "r") as jsonfile:
            ooddict = json.load(jsonfile)
        plot_all_ens_de_mean_cal_recal(ooddict, caldict, i)
