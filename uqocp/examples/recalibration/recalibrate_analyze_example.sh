#!/bin/bash

python ../../utils/main_uq_analysis.py \
--val_unc_path ../../examples/latent_distance_index/eq2gnoc_per_atom_sum_is2re_train_67%/eq2gnoc_per_atom_sum_is2re_train_67%_id.npz \
--test_unc_path ../../examples/latent_distance_index/eq2gnoc_per_atom_sum_is2re_train_67%/eq2gnoc_per_atom_sum_is2re_train_67%_ood.npz \
--units eV \
--uncertainty_method latent_distance \
--savefig_path figures/example \
--label sum \