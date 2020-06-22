# kineticmodel_bdi
Bayesian data integration for kinetic models written in Python.
In our [publication in ACS Omega](https://pubs.acs.org/doi/10.1021/acsomega.0c01109) we introduce the theory of our approach and evaluate different reaction mechanisms for the acceleration of trypsin autolysis caused by silica surfaces.

# Usage / Tutorial

Interested users can reproduce the proof of concept from the paper. 
The necessary jupyter notebooks can be found in `tutorial_and_POC/`. 
It demonstrates most options.
Users can easily modify the kinetic models shown there to their needs.

1. The synthetic data are generated in `generate_observed_data.ipynb`. 
This notebook is just for reference, you can find the synthetic data as shown in the SI of the paper in ` 	observed_kin_data.csv, observed_eq_adsorption_data.csv, observed_td_adsorption_data.csv`.

2. The definition of the kinetic models, priors etc. is found in `infer_parameters.ipynb`.
The inference itself is also done within this notebook.
Running all of it can take several hours or up to 1-2 days on your desktop machine.
Upon completion, you will find the MCMC chains stored in `.h5` files. 
These are relatively large (up to a few hundred MB) so we don't share them via GitHub.
If you face a problem at this stage, feel free to contact me and I will share my results with you.
`LH_infer_parameters.ipynb` is analogous but with slightly different settings and priors for the LH mechanism.

3. The results from the inference are analysed in `analysis.ipynb`.
It contains visualizations of the posterior predictive checks, marginalized posterior distributions and the other figures from the SI of the paper concerned with the proof of concept.

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

* [`chempy`](https://github.com/bjodah/chempy) (create ODE systems)
* [`pyodesys`](https://github.com/bjodah/pyodesys) (solve ODE systems)
* [`emcee`](https://github.com/dfm/emcee) (MCMC sampling)
* [`pymc`](https://github.com/pymc-devs/pymc) (Gelman-Rubin diagnostics)
* [`pytables`](http://www.pytables.org/) (h5 files manipulation)

Only for parallelization:
* [`dill`](https://github.com/uqfoundation/dill)
* [`pathos`](https://github.com/uqfoundation/pathos)

For you convenience, you can create a new conda environment with all necessary dependencies from `environment_p2.yml`.
