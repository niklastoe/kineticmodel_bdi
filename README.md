# kineticmodel_bdi
Bayesian data integration for kinetic models written in Python

# Usage / Tutorial

Interested users can reproduce the proof of concept from the paper. 
The necessary jupyter notebooks can be found in `tutorial_and_POC/`. 
It demonstrates most options.
Users can easily modify the kinetic models shown there to their needs.

# Compatibility with Python3

This package was developed and used with Python2. 
Due to the end of life of Python2 by the end of 2019, I ensured compatability with Python3 as well.

## Compilation of native code fails with `CVODE`

On my local machine, Python3 would not work with `CVODE` (which was used to obtain the results in the paper).
If you experience the same problem, try using `GSL` or `ODEINT`.
These worked for me using Python3 but (in my limited experience) were substantially slower than `CVODE`. 
If you can fix the issue with `CVODE` and Python3, please create a pull request.

## Parallel computing with Pathos

Under Python2, Pathos can be used to effortlessly parallelize the sampling.
With Python3, parralelization is currently not possible.

# Installation

Download the code/clone the repository. Don't forget to add the path to your copy to your PYTHONPATH.

## Dependencies (only those not included in an anaconda installation by default)

* chempy (create ODE systems)
* pyodesys (solve ODE systems)
* emcee (MCMC sampling)
* pymc (Gelman-Rubin diagnostics)
* pytables (h5 files manipulation)

Only for parallelization:
* dill
* pathos

For you convenience, you can create a new conda environment with all necessary dependencies from `environment_p2.yml`.
