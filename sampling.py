import copy
import emcee
import numpy as np
import json
import pandas as pd

from workflows.usability import create_parameter_dictionary_for_function, identify_necessary_parameters


class SamplingEnvironment(object):

    def __init__(self, prior_distribution_dict, logp_dict, reformatting_function=None):
        self.prior_distributions = prior_distribution_dict

        self.reformat = reformatting_function

        self.logp_func_parameters = logp_factory(copy.deepcopy(logp_dict), self.log_prior)

        # after how many steps to check for convergence
        self.convergence_check_interval = 1000
        # how many samples we want at least
        self.min_iid = 50
        # max deviation between old_tau and tau to be considered converged
        self.convergence_threshold = 0.01

    def log_prior(self, **kwargs):
        """return sum of log priors for a dictionary of prior_functions"""

        if self.reformat is not None:
            formatted_parameters = self.reformat(kwargs, return_ds=False)
        else:
            formatted_parameters = kwargs

        prior = 0
        for sel_parameter in formatted_parameters.keys():
            if sel_parameter in self.prior_distributions.keys():
                prior += self.prior_distributions[sel_parameter].logpdf(formatted_parameters[sel_parameter])

        return prior

    def random_start_positions(self):
        """generate random start positions which are not impossible i.e. have -np.inf probability"""
        current_score = -np.inf
        while ~np.isfinite(current_score):
            required_parameters = self.logp_func_parameters.required_parameters
            random_parameters = {x: float(self.prior_distributions[x].rvs()) for x in required_parameters}

            # calculate log_posterior
            current_score = self.logp_func_parameters(**random_parameters)

            # if logp_func passes more than just logp (e.g. dictionary to store as blob in emcee),
            # check what is the logp and use that!!
            if type(current_score) == tuple:
                current_score = current_score[0]

        return random_parameters

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


def run_logp(logp_func, parameters):
    """evaluate logp for parameters without having to select parameters first"""

    appropriate_parameters = create_parameter_dictionary_for_function(logp_func, dict(parameters))
    return logp_func(**appropriate_parameters)


def evaluate_multiple_logp(dict_of_logps, parameters):
    """evaluate parameters for all given logp functions"""
    logps = {}

    available_funcs = dict_of_logps.keys()

    prior_func = [x for x in available_funcs if 'prior' in x]
    if len(prior_func) == 1:
        curr_prior = dict_of_logps[prior_func[0]](**parameters)
        logps['prior'] = curr_prior
    elif len(prior_func) == 0:
        curr_prior = 0
    else:
        raise ValueError('More than one prior function passed!')

    likelihood_funcs = [x for x in available_funcs if x not in prior_func]

    # don't waste time calculating the likelihood if prior is prohibitive
    if np.isfinite(curr_prior):
        for x in likelihood_funcs:
            logps[x] = run_logp(dict_of_logps[x], parameters)
        sum_logp = sum(logps.values())
    else:
        for x in likelihood_funcs:
            logps[x] = np.nan
        sum_logp = curr_prior

    # return the sum of logps and the dictionary
    # also return empty string, otherwise emcee runs into a weird error trying to handle single blob
    return sum_logp, json.dumps(logps), ''


def logp_factory(dict_of_logps, incl_prior=None):
    """return a function that evaluates the sum of logps for all inputs"""

    if incl_prior:
        dict_of_logps['prior'] = incl_prior

    def logp_from_factory(**kwargs):
        return evaluate_multiple_logp(dict_of_logps, kwargs)

    required_parameters = identify_necessary_parameters(dict_of_logps.values())
    logp_from_factory.required_parameters = [x for x in required_parameters if x != 'self']

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


def sample_until_convergence(sampler, nsteps, starting_pos, thin_by=1):
    """sample for nsteps, stop if autocorrelation time tau converges beforehand"""
    # Check for convergence, taken from https://emcee.readthedocs.io/en/latest/tutorials/monitor/
    # We'll track how the average autocorrelation time estimate changes

    # after how many steps to check for convergence
    convergence_check_interval = 2500
    # how many samples we want at least
    min_iid = 50
    # max deviation between old_tau and tau to be considered converged
    convergence_threshold = 0.02

    index = 0
    autocorr = []

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
        autocorr.append(np.mean(tau))
        index += 1

        # store in h5 file
        tau_df = pd.DataFrame(tau.reshape(1,-1),
                              columns=sampler.parm_names,
                              index=[sampler.iteration])
        tau_df.to_hdf(sampler.backend.filename, 'autocorrelation', format='table', append=True)

        # Check convergence
        converged = np.all(tau * min_iid < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < convergence_threshold)
        if converged:
            break
        old_tau = tau

    autocorr = pd.Series(autocorr, index=[i * convergence_check_interval for i in range(len(autocorr))])
    sampler.autocorr_history = autocorr
