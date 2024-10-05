#!/bin/bash

python ../../utils/main_uq_analysis.py \
--val_unc_path ../../../data/latent_distance_results/eq2_inv_l4_edges/eq2inv_unrotated_middle_per_atom_mean_is2re_train_id.npz  \
--test_unc_path ../../../data/latent_distance_results/eq2_inv_l4_edges/eq2inv_unrotated_middle_per_atom_mean_is2re_train_id.npz  \
--units eV \
--uncertainty_method latent_distance \
--savefig_path figures/eq2 \
--label eq2_inv_l4_edges \