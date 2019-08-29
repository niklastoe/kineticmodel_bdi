import unittest as ut
import xarray as xr

from workflows.kinetic_modeling.test import test_KineticModel
from workflows.kinetic_modeling.bayesian_framework import Likelihood, OrdinaryStandardDeviation


class TestLikelihood(test_KineticModel.TestKineticModelFirst):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super(TestLikelihood, self).__init__(*args, **kwargs)

        self.model.create_native_odesys()
        self.sel_std_dev = 1e-7
        self.likelihood_ordinary_obj = Likelihood(self.model, OrdinaryStandardDeviation(self.sel_std_dev))

    def test_likelihood_creation(self):
        self.assertTrue(isinstance(self.likelihood_ordinary_obj, Likelihood))

    def test_calc_likelihood(self):
        """check that input parameters yield maximum likelihood"""
        likelihood = self.likelihood_ordinary_obj.calc_likelihood(self.get_parameters())
        self.assertAlmostEqual(likelihood, self.likelihood_ordinary_obj.max_likelihood, delta=1e-8)

    def test_evaluation(self):
        """Sampling should allow to recover input std_dev and mu (=y_model=y_exp in this case)"""

        samples = []

        for x in range(100):
            samples.append(self.likelihood_ordinary_obj.evaluate_parameters(self.get_parameters()))

        samples = xr.concat([df.to_xarray() for df in samples], "samples")

        mu = samples.mean(dim='samples').to_dataframe() - self.model.exp_data
        std = samples.std(dim='samples').to_dataframe()
        self.assertAlmostEqual(mu.mean().mean(), 0, delta=self.model.exp_data.max().max() * 5e-3)
        self.assertAlmostEqual(std.mean().mean(), self.sel_std_dev, delta=self.sel_std_dev*0.1)

if __name__ == '__main__':
    ut.main(verbosity=2)
