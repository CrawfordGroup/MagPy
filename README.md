MagPy
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/CrawfordGroup/MagPy/workflows/CI/badge.svg)](https://github.com/CrawfordGroup/magpy/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/CrawfordGroup/MagPy/graph/badge.svg?token=SN87ODLNBW)](https://codecov.io/gh/CrawfordGroup/MagPy)

A Python reference implementation for including explicit magnetic fields in quantum chemical
calculations. Current capabilities include:
  - Complex Hartree-Fock
  - Atomic axial tensors via numerical derivatives of wave functions

This repository is currently under development. To do a developmental install, download this repository and type `pip install -e .` in the repository directory.

This package requires the following:
  - [psi4](https://psicode.org)
  - [numpy](https://numpy.org/)
  - [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/)

### Copyright

Copyright (c) 2023, T. Daniel Crawford


#### Acknowledgements

Project structure based on the
[MolSSI's](https://molssi.org) [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms)
version 1.1.
