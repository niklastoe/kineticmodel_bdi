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

if __name__ == '__main__':
    ut.main(verbosity=2)
