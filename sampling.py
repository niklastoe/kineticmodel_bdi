import copy
import emcee
import numpy as np
import pandas as pd

from kineticmodel_bdi.bayesian_framework import find_necessary_parameters, Likelihood
from kineticmodel_bdi.analysis import read_last_autocorrelation
from dill import PicklingError


def dummy_reformatting_function(parameters):
    """this function does nothing, it simply returns the input dictionary
    it is necessary as a dummy reformatting_function (if you don't want/need any reformatting)"""
    return parameters


class SamplingEnvironment(object):
    """create everything that's necessary for sampling:
    log_prior function
    posterior function [evaluates prior and all passed likelihoods (from passed objects)]
    emcee sampler
    start positions for emcee sampler"""

    def __init__(self,
                 prior_distribution_dict,
                 likelihood_object_dict,
                 reformatting_function=dummy_reformatting_function):
        self.prior_distributions = prior_distribution_dict
        self.reformat = reformatting_function

        self.logp_func_parameters = logp_factory(likelihood_object_dict, self.reformat, self.log_prior)

        self.required_parameters = find_necessary_parameters(self.logp_func_parameters, {'ignore_prior': True})

        # after how many steps to check for convergence
        self.convergence_check_interval = 1000
        # how many samples we want at least
        self.min_iid = 50
        # max deviation between old_tau and tau to be considered converged
        self.convergence_threshold = 0.01

    def log_prior(self, parameters):
        """return sum of log priors for a dictionary of prior_functions"""

        prior = 0
        for sel_parameter in parameters.keys():
            if sel_parameter in self.prior_distributions.keys():
                prior += self.prior_distributions[sel_parameter].logpdf(parameters[sel_parameter])

        return prior

    def random_start_positions(self):
        """generate random start positions which are not impossible i.e. have -np.inf probability"""
        current_score = -np.inf
        while ~np.isfinite(current_score):
            random_parameters = {x: float(self.prior_distributions[x].rvs()) for x in self.required_parameters}

            # calculate log_posterior
            current_score = self.logp_func_parameters(random_parameters)

            # if logp_func passes more than just logp (e.g. dictionary to store as blob in emcee),
            # check what is the logp and use that!!
            if type(current_score) == tuple:
                current_score = current_score[0]

        return random_parameters

    def setup_sampler(self, nwalkers, filename=None, sel_pool=None):

        if sel_pool is not None:
            if self.logp_func_parameters.Likelihood_instance_used_directly:
                raise PicklingError("You cannot use Likelihood.calc_likelihood directly if you want to use Pool! "
                                    "You need to define a function in your notebook for every instance of Likelihood"
                                    " that calls Likelihood.calc_likelihood. "
                                    "It's a bit stupid but I don't know how to avoid that! \n"
                                    "The code looks like this: \n"
                                    "\n"
                                    "dict_of_functions = {} \n"
                                    "for key in dict_of_likelihood_objects: \n"
                                    "   def function_wrapper(parameters): \n"
                                    "       return dict_of_likelihood_objects[key].calc_likelihood(parameters) \n"
                                    "   function_wrapper.__name__ = key + '_likelihood_func_wrapper' \n"
                                    "   dict_of_functions[key] = function_wrapper \n")

        my_parm_dict = self.random_start_positions()
        my_parms = list(my_parm_dict.keys())

        def update_parameter_dict(theta):
            return {my_parms[idx]: x for idx, x in enumerate(theta)}

        def logp_func_theta(theta):
            """calculate logp_func based on array of parameters, not dictionary"""
            curr_parms = update_parameter_dict(theta)
            return self.logp_func_parameters(curr_parms)

        ndims = len(my_parms)

        if filename is not None:
            backend = emcee.backends.HDFBackend(filename)

            # store parameter names
            parameter_names_ds = pd.Series([0] * len(my_parms), index=my_parms)
            parameter_names_ds.to_hdf(backend.filename, 'parameter_names', format='fixed')

            # store blob names
            blob_names_ds = pd.Series([0] * len(self.logp_func_parameters.names), index=self.logp_func_parameters.names)
            blob_names_ds.to_hdf(backend.filename, 'blob_names', format='fixed')

        else:
            backend = None
        sampler = emcee.EnsembleSampler(nwalkers, ndims,
                                        logp_func_theta,
                                        backend=backend,
                                        pool=sel_pool)

        sampler.parm_names = my_parms

        return sampler

    def resume_positions_or_create_new_ones(self, sampler):
        """try to restart from the previous state, otherwise use random new starting positions"""

        # if there are previous steps, continue and overwrite generated starting_pos
        try:
            starting_pos = sampler.get_last_sample().coords
        except AttributeError:
            starting_pos = np.array([list(self.random_start_positions().values()) for x in range(sampler.nwalkers)])

        return starting_pos


def evaluate_multiple_likelihoods(dict_of_functions, formatted_parameters, curr_prior=0):
    """evaluate parameters for all given logp functions"""
    logps = {}
    # don't waste time calculating the likelihood if prior is prohibitive
    if np.isfinite(curr_prior):
        for x in dict_of_functions:
            logps[x] = dict_of_functions[x](formatted_parameters)
        sum_logp = sum(logps.values())
    # if prior is prohibitive, set likelihoods to np.nan
    else:
        for x in dict_of_functions:
            logps[x] = np.nan
        # set prior to the calculated value
        logps['prior'] = curr_prior
        # cannot return np.nan to emcee
        sum_logp = curr_prior

    # return the sum of logps and the dictionary
    # also return empty string, otherwise emcee runs into a weird error trying to handle single blob
    return tuple([sum_logp] + list(logps.values()))


def logp_factory(dict_of_likelihood_objects, reformat_func, prior_function=None):
    """return a function that evaluates the sum of logps for all inputs"""

    # confirm if all entries in dict are indeed instances of Likelihood
    if np.array([type(x) == Likelihood for x in dict_of_likelihood_objects.values()]).all():
        dict_of_functions = {x: dict_of_likelihood_objects[x].calc_likelihood for x in dict_of_likelihood_objects}
        Likelihood_instance_used_directly = True
    else:
        dict_of_functions = dict_of_likelihood_objects
        Likelihood_instance_used_directly = False

    if prior_function:
        dict_of_functions['prior'] = prior_function

    def logp_from_factory(parameters, ignore_prior=False):
        formatted_parameters = reformat_func(copy.deepcopy(parameters))
        if ignore_prior:
            curr_prior = 0
        else:
            curr_prior = dict_of_functions['prior'](formatted_parameters)

        return evaluate_multiple_likelihoods(dict_of_functions, formatted_parameters, curr_prior)

    logp_from_factory.Likelihood_instance_used_directly = Likelihood_instance_used_directly

    logp_from_factory.names = list(dict_of_functions.keys())

    return logp_from_factory


class UniformMinMax(object):

    def __init__(self, val_min, val_max):
        self.min = val_min
        self.max = val_max

    def logpdf(self, value):
        if self.min < value < self.max:
            return 0.0
        else:
            return -np.inf

    def rvs(self):
        return self.min + (self.max - self.min) * np.random.rand()


def sample_until_convergence(sampler, nsteps, starting_pos,
                             thin_by=1,
                             convergence_check_interval=2500):
    """sample for nsteps, stop if autocorrelation time tau converges beforehand"""
    # Check for convergence, taken from https://emcee.readthedocs.io/en/latest/tutorials/monitor/
    # We'll track how the average autocorrelation time estimate changes

    # how many samples we want at least
    min_iid = 50
    # max deviation between old_tau and tau to be considered converged
    convergence_threshold = 0.02

    autocorrelation = []

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(starting_pos,
                                 thin_by=thin_by,
                                 iterations=nsteps,
                                 progress=True):
        # Only check convergence every n steps
        if sampler.iteration % (convergence_check_interval / thin_by):
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)

        # store in h5 file if one is written
        tau_df = pd.DataFrame(tau.reshape(1, -1),
                              columns=sampler.parm_names,
                              index=[sampler.iteration])
        if hasattr(sampler.backend, 'filename'):
            tau_df.to_hdf(sampler.backend.filename, 'autocorrelation', format='table', append=True)

        autocorrelation.append(tau_df)

        # Check convergence
        converged = np.all(tau * min_iid < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < convergence_threshold)
        if converged:
            break
        old_tau = tau

    if len(autocorrelation) > 0:
        sampler.autocorrelation = pd.concat(autocorrelation)


def read_in_sampler(h5_file):
    """read in sampler from h5 file and return it
    note: logp function is just a dummy, you can't sample with it!!"""
    backend = emcee.backends.HDFBackend(h5_file)
    sampler = emcee.EnsembleSampler(backend.shape[0],
                                    backend.shape[1],
                                    dummy_reformatting_function,
                                    args=(),
                                    backend=backend)

    try:
        sampler.parm_names = read_last_autocorrelation(sampler).index
    except KeyError:
        pass
    return sampler
