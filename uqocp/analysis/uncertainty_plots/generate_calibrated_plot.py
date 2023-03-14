import sys
sys.path.insert(0, "/home/jovyan/working/uncertainty-toolbox/uncertainty_toolbox")
import metrics_calibration
import recalibration

import json
import numpy as np
import matplotlib.pyplot as plt

import time
start_time = time.time()
print("starting")
print("--- %s seconds ---" % (time.time() - start_time))



# --------------------------------------------------------------------------------------------------------
# uncalibrated id calibration plot ------------------------------------
# --------------------------------------------------------------------------------------------------------

with open("/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/id_errors.json", "r") as jsonfile:
    udict = json.load(jsonfile)

predicted_y = np.array(udict["en_error"])
stdev = np.array(udict["en_stdev"])
true_y = np.array([0 for i in predicted_y])

# import uncertainty_toolbox as ut
p_exp, p_obs = (p.flatten() for p in metrics_calibration.get_proportion_lists_vectorized(
    predicted_y, # y predicted
    stdev, # y predicted standard deviation (uncertainty)
    true_y, # y true value
      prop_type='interval',
      num_bins=100))

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
fig.savefig("uncalibrated_id_calibration_plot.png", dpi=300)
print("generated id calibration plot")
print("--- %s seconds ---" % (time.time() - start_time))


# --------------------------------------------------------------------------------------------------------
# calibrated id calibration plot ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
std_recalibrator = recalibration.get_std_recalibrator(
    predicted_y, # y predicted
    stdev, # y predicted standard deviation (uncertainty)
    true_y, # y true value
)
print("finished recalibrator")
print("--- %s seconds ---" % (time.time() - start_time))

print("calibrator std recal ratio: " +str(std_recalibrator(1)))

stdev_recal = std_recalibrator(stdev)

p_exp, p_obs = (p.flatten() for p in metrics_calibration.get_proportion_lists_vectorized(
    predicted_y, # y predicted
    stdev_recal, # y predicted standard deviation (uncertainty)
    true_y, # y true value
    prop_type='interval',
    num_bins=100))
print("finished recalibration")
print("--- %s seconds ---" % (time.time() - start_time))

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
fig.savefig("id_calibrated_id_calibration_plot.png", dpi=300)
print("generated calibrated id calibration plot")
print("--- %s seconds ---" % (time.time() - start_time))



# --------------------------------------------------------------------------------------------------------
# uncalibrated ood calibration plot ------------------------------------
# --------------------------------------------------------------------------------------------------------

with open("/home/jovyan/shared-scratch/joe/jobs/uncertainty/uqocp/uqocp/analysis/ood_errors_5gnoc_only.json", "r") as jsonfile:
    udict = json.load(jsonfile)

predicted_y = np.array(udict["en_error"])
stdev = np.array(udict["en_stdev"])
true_y = np.array([0 for i in predicted_y])

p_exp, p_obs = (p.flatten() for p in metrics_calibration.get_proportion_lists_vectorized(
    predicted_y, # y predicted
    stdev, # y predicted standard deviation (uncertainty)
    true_y, # y true value
      prop_type='interval',
      num_bins=100))

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
fig.savefig("uncalibrated_ood_calibration_plot.png", dpi=300)
print("generated uncalibrated ood calibration plot")
print("--- %s seconds ---" % (time.time() - start_time))



# --------------------------------------------------------------------------------------------------------
# calibrated ood calibration plot ------------------------------------
# --------------------------------------------------------------------------------------------------------

stdev_recal = std_recalibrator(stdev)
print('recalibrated ood uq predictions with id recalibration')
print("--- %s seconds ---" % (time.time() - start_time))

p_exp, p_obs = (p.flatten() for p in metrics_calibration.get_proportion_lists_vectorized(
    predicted_y, # y predicted
    stdev_recal, # y predicted standard deviation (uncertainty)
    true_y, # y true value
    prop_type='interval',
    num_bins=100))

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
fig.savefig("id_calibrated_ood_calibration_plot.png", dpi=300)
print("generated ood calibration plot, calibrated with id calibration")
print("--- %s seconds ---" % (time.time() - start_time))