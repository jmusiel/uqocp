Uncertainty quantification methods and tools for OCP models

## UQOCP
This repository provides examples and implementations of the methods and experiments found in the paper "Improved Uncertainty Estimation of Graph Neural Network Potentials Using Engineered Latent Space Distances". The manuscript is up on arXiv [here](https://arxiv.org/abs/2407.10844).

## Overview
Examples for how to use the latent distance method described in the paper, (generating the index, recalibrating, predicting uncertainty on the platinum coverage example) are found in [`uqocp/examples`](uqocp/examples/).

The latent distance extraction modifications for each of the model architectures tested in the paper can be found in [`uqocp/ocp_latent`](uqocp/ocp_latent/), these model implementations require [OCP checkpoints](https://github.com/FAIR-Chem/fairchem/blob/3012925adc1d1273b2ab394c2d5274cce9698b0f/docs/core/model_checkpoints.md#L4), and the ocpmodels repository as described in the requirements section below.

Utility functions used in for the experiments in the manuscript for evaluating calibration, running recalibration, generating the figures, and other miscellaneous tasks such as loading data can be found in [`uqocp/utils`](uqocp/utils/).

Implementations of the latent distance method, and the other uncertainty quantification methods are found in [`uqocp/uncertainty_quantification`](uqocp/uncertainty_quantification/).

## Installation
Clone the repository, navigate to the root of the directory, and run:
```pip install .```

## Requirements
- Python (>= 3.8)

- FAISS (>= 1.7)

- The latent distance implementations in this repository were written to work with [this](https://github.com/FAIR-Chem/fairchem/tree/c52aeeacb3854c8d7841ab3953a9cfef284a301f) commit of the ocpmodels repository.

_for this version of ocpmodels_
- torch (= 1.11)
- torch-geometric (= 2.1)

_for GPU accelerated inference_
- CUDA (>= 10.2)

_Other versions of NumPy should work, but this one was used to create the .npz files_
- NumPy (= 1.23)

## Data availability
All train, validation, calibration and test data for the RS2RE task was derived from OC20 IS2RE.
- FAISS index data for the latent distance method used all structures from the [IS2RE training dataset](https://fair-chem.github.io/core/datasets/oc20.html#initial-structure-to-relaxed-structure-is2rs-and-initial-structure-to-relaxed-energy-is2re-tasks) relaxed to a max force of 0.03 eV/Å with the [Equiformer V2 153M checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt).
- Calibration data is derived from the [IS2RE validation in-domain (val id) dataset](https://fair-chem.github.io/core/datasets/oc20.html#initial-structure-to-relaxed-structure-is2rs-and-initial-structure-to-relaxed-energy-is2re-tasks), relaxed in the same way.
- Test data is derived from the [IS2RE validation out-of-domain (val ood) dataset](https://fair-chem.github.io/core/datasets/oc20.html#initial-structure-to-relaxed-structure-is2rs-and-initial-structure-to-relaxed-energy-is2re-tasks), relaxed in the same way.

The latent distance results generated for EqV2 and GemnetOC can be found in [`data/latent_distance_results`](data/latent_distance_results/). 

Samples of each data set are included in this repository at [`data`](data/) for running the examples. However methods like latent distance rely on a more complete datset to work well, therefore using it on this example data may not be representative of performance on the whole data set. If you would like access to a FAISS latent distance for one of the checkpoints generated for this manuscript, please contact Joe Musielewicz or John Kitchin directly, since they are quite large (~10 GB or more depending on the size of the latent vectors) and difficult to host.

## Citation
If you used UQOCP in your work, we would appreciate you citing our [manuscript](https://arxiv.org/abs/2407.10844)!
```
Musielewicz, J.; Lan, J.; Uyttendaele, M.; Kitchin, J. Improved Uncertainty Estimation of Graph Neural Network Potentials Using Engineered Latent Space Distances. _arXiv_ **2024**, abs/2407.10844.
```
_bibtex:_
```bibtex
@misc{musielewicz2024improveduncertaintyestimationgraph,
      title={Improved Uncertainty Estimation of Graph Neural Network Potentials Using Engineered Latent Space Distances}, 
      author={Joseph Musielewicz and Janice Lan and Matt Uyttendaele and John R. Kitchin},
      year={2024},
      eprint={2407.10844},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.10844}, 
}
```