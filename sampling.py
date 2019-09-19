import copy
import emcee
import numpy as np
import json
import pandas as pd


def pass_dict(parameters):
    """this function does nothing, it simply returns the input dictionary"""
    return parameters


class SamplingEnvironment(object):

    def __init__(self, prior_distribution_dict, logp_dict, reformatting_function=pass_dict):
        self.prior_distributions = prior_distribution_dict

        self.reformat = reformatting_function

        self.logp_func_parameters = logp_factory(logp_dict, self.reformat, self.log_prior)

        self.required_parameters = self.find_necessary_parameters()

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

    def find_necessary_parameters(self):
        """return a list of necessary parameters for a function that takes one dictionary as input"""

        success = False
        curr_dict = {}
        # catch KeyErrors until likelihood can be evaluated properly
        while not success:
            try:
                self.logp_func_parameters(curr_dict, ignore_prior=True)
                success = True
            except KeyError, e:
                missing_parm = e[0]
                curr_dict[missing_parm] = self.prior_distributions[missing_parm].rvs()

        return curr_dict.keys()

    def setup_sampler(self, nwalkers, filename=None, sel_pool=None):
        my_parm_dict = self.random_start_positions()
        my_parms = my_parm_dict.keys()

        def update_parameter_dict(theta):
            return {my_parms[idx]: x for idx, x in enumerate(theta)}

        def logp_func_theta(theta):
            """calculate logp_func based on array of parameters, not dictionary"""
            curr_parms = update_parameter_dict(theta)
            return self.logp_func_parameters(**curr_parms)

        ndims = len(my_parms)

        if filename is not None:
            backend = emcee.backends.HDFBackend(filename)
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

        starting_pos = None
        # if there are previous steps, continue and overwrite generated starting_pos
        try:
            starting_pos = sampler.get_last_sample().coords
        except AttributeError:
            starting_pos = np.array([self.random_start_positions().values() for x in range(sampler.nwalkers)])

        return starting_pos


def pymc_logp_val(val, dist):
    return float(dist.logp(val).eval())


def evaluate_multiple_likelihoods(dict_of_functions, parameters, reformat_func, curr_prior=0):
    """evaluate parameters for all given logp functions"""
    logps = {}

    formatted_parameters = reformat_func(parameters)

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
    return sum_logp, json.dumps(logps), ''


def logp_factory(dict_of_likelihood_objects, reformat_func, prior_function=None):
    """return a function that evaluates the sum of logps for all inputs"""

    dict_of_functions = {x: dict_of_likelihood_objects[x].calc_likelihood for x in dict_of_likelihood_objects}

    if prior_function:
        dict_of_functions['prior'] = prior_function

    def logp_from_factory(parameters, ignore_prior=False):
        if ignore_prior:
            curr_prior = 0
        else:
            curr_prior = dict_of_functions['prior'](reformat_func(parameters))

        return evaluate_multiple_likelihoods(dict_of_functions, parameters, reformat_func, curr_prior)

    return logp_from_factory

class uniform_minmax(object):

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
        tau_df = pd.DataFrame(tau.reshape(1,-1),
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
