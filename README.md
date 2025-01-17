# Revisiting Data Augmentation for Ultrasound Images

This repository contains the official source code for our article Revisiting Data Augmentation for Ultrasound Images.

## Organization

Our source code is organized into three directories:

### `ultrabench`

The command line application for UltraBench (to be published on PyPI) for preparing the datasets and tasks supported by the benchmark. See the enclosed `README` for more details.

### `usaugment`

A standalone Python package (to be published on PyPI) for the ultrasound-specific augmentations implemented and evaluated in the paper (Depth Attenuation, Gaussian Shadow, Haze Artifact and Speckle Reduction). See the enclosed `README` for more details.

### `usaugment-experiments`

The code and documentation for running the experiments presented in the paper and reproducing the results.
