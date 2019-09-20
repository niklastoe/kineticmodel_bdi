import copy
import unittest as ut
from workflows.kinetic_modeling.test import test_Likelihood
from workflows.kinetic_modeling.sampling import SamplingEnvironment, UniformMinMax

class TestSamplingEnvironment(test_Likelihood.TestLikelihoodFunction):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super(TestSamplingEnvironment, self).__init__(*args, **kwargs)

        prior_m = UniformMinMax(-1, 1)
        prior_b = UniformMinMax(0, 2)
        prior_distributions = {'m': prior_m,
                               'b': prior_b}

        likelihood_object_dict = {'A': self.likelihood_ordinary_obj,
                                  'B': self.likelihood_ordinary_obj}

        self.env = SamplingEnvironment(prior_distributions, likelihood_object_dict)

        def reformat_parameters(parameters):
            formatted_parameters = copy.deepcopy(parameters)
            formatted_parameters['b'] = parameters['c'] + 1
            return formatted_parameters

        self.env_reformatted = SamplingEnvironment(prior_distributions, likelihood_object_dict, reformat_parameters)

    def test_required_parameters(self):
        """are all required parameters detected correctly?"""
        theta = set(['m', 'b'])
        theta_reformatted = set(['m', 'c'])

        self.assertEqual(theta, set(self.env.required_parameters))
        self.assertEqual(theta_reformatted, set(self.env_reformatted.required_parameters))

    def test_posterior_function(self):
        posterior_evaluation = self.env.logp_func_parameters(self.get_parameters())
        logp, blobs = posterior_evaluation[0], posterior_evaluation[1:]
        # since the parameters are the true ones (i.e. those that generated the data), logp should be very close to max
        self.assertAlmostEqual(logp, 2*self.likelihood_ordinary_obj.max_likelihood)
        # confirm that blobs are one string containing json and and empty string (weird thing by emcee)
        self.assertEqual(type(blobs[0]), str)
        self.assertEqual(blobs[1], '')

if __name__ == '__main__':
    ut.main(verbosity=2)
