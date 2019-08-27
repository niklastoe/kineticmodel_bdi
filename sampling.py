from functools import partial
import numpy as np
import json

from workflows.usability import create_parameter_dictionary_for_function, identify_necessary_parameters


class SamplingEnvironment(object):

    def __init__(self, prior_distribution_dict, reformatting_function=None):
        self.prior_distributions = prior_distribution_dict
        for x in self.prior_distributions:
            self.prior_distributions[x].logp_val = partial(pymc_logp_val, dist=self.prior_distributions[x])

        self.reformat = reformatting_function

    def log_prior(self, parameters):
        """return sum of log priors for a dictionary of prior_functions"""

        if self.reformat is not None:
            formatted_parameters = self.reformat(parameters, return_ds=False)
        else:
            formatted_parameters = parameters

        prior = 0
        for sel_parameter in formatted_parameters.keys():
            if sel_parameter in self.prior_distributions.keys():
                prior += self.prior_distributions[sel_parameter].logp_val(parameters[sel_parameter])

        return prior

    def random_start_positions(self, logp_func):
        """generate random start positions which are not impossible i.e. have -np.inf probability"""
        current_score = -np.inf
        while ~np.isfinite(current_score):
            required_parameters = identify_necessary_parameters([logp_func])
            test_parameters = {x: float(self.prior_distributions[x].random()) for x in required_parameters}

            # calculate log_posterior
            current_score = logp_func(**test_parameters)

            # if logp_func passes more than just logp (e.g. dictionary to store as blob in emcee),
            # check what is the logp and use that!!
            if type(current_score) == tuple:
                current_score = current_score[0]

        return test_parameters


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
        curr_prior = dict_of_logps[prior_func[0]](parameters)
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


def logp_factory(dict_of_logps):
    """return a function that evaluates the sum of logps for all inputs"""

    def logp_from_factory(**kwargs):
        return evaluate_multiple_logp(dict_of_logps, kwargs)

    logp_from_factory.required_parameters = identify_necessary_parameters(dict_of_logps.values())
    return logp_from_factory
