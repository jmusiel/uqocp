#!/bin/bash

python ../../utils/main_uq_analysis.py \
--val_unc_path ../../../data/latent_distance_results/gnoc/eq2gnoc_per_atom_sum_is2re_train_id.npz \
--test_unc_path ../../../data/latent_distance_results/gnoc/eq2gnoc_per_atom_sum_is2re_train_ood.npz \
--units eV \
--uncertainty_method latent_distance \
--savefig_path figures/gnoc \
--label gnoc \