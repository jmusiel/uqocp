#!/bin/bash

python ../../distance/conformal_prediction.py \
--train_dir \
/home/jovyan/working/repos/uqocp/data/train \
--calib_dir \
/home/jovyan/working/repos/uqocp/data/calib \
--test_dir \
/home/jovyan/working/repos/uqocp/data/test \
--per_atom sum \
--debug n \
--index_constructor IVF65536_HNSW32,Flat \
--save_npz eq2gnoc_per_atom_sum_is2re_train_%.npz \
--save_dir eq2gnoc_per_atom_sum_is2re_train_% \
--load_index_dir /private/home/jmu/working/repos/joe_fair_dev/joedev/distance/back_to_cmu/slurm_outputs_eq2_nll_calibration/gnoc/eq2gnoc_per_atom_sum_is2re_train_68% \