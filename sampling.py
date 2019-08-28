import numpy as np
import json
from scipy.stats import uniform

from workflows.usability import create_parameter_dictionary_for_function, identify_necessary_parameters


class SamplingEnvironment(object):

    def __init__(self, prior_distribution_dict, logp_dict, reformatting_function=None):
        self.prior_distributions = prior_distribution_dict

        self.reformat = reformatting_function

        self.logp_func_parameters = logp_factory(logp_dict, self.log_prior)

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

def uniform_minmax(val_min, val_max):
    loc = val_min
    scale = val_max - val_min
    return uniform(loc, scale)
