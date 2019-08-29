import unittest as ut

from workflows.kinetic_modeling.test import test_KineticModel
from workflows.kinetic_modeling.bayesian_framework import Likelihood, OrdinaryStandardDeviation


class TestLikelihood(test_KineticModel.TestKineticModelFirst):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super(TestLikelihood, self).__init__(*args, **kwargs)

        self.model.create_native_odesys()
        self.likelihood_ordinary_obj = Likelihood(self.model, OrdinaryStandardDeviation(1e-7))

    def test_likelihood_creation(self):
        self.assertTrue(isinstance(self.likelihood_ordinary_obj, Likelihood))

    def test_calc_likelihood(self):
        """check that input parameters yield maximum likelihood"""
        likelihood = self.likelihood_ordinary_obj.calc_likelihood(self.get_parameters())
        self.assertAlmostEqual(likelihood, self.likelihood_ordinary_obj.max_likelihood, delta=1e-8)

if __name__ == '__main__':
    ut.main(verbosity=2)
