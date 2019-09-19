import numpy as np
import pandas as pd
import unittest as ut
import xarray as xr

from workflows.kinetic_modeling.test import test_KineticModel
from workflows.kinetic_modeling.bayesian_framework import Likelihood, OrdinaryStandardDeviation


class TestLikelihoodBase(test_KineticModel.TestKineticModelFirst):
    __test__ = False

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

        y_exp = self.likelihood_ordinary_obj.evaluate_parameters(self.get_parameters(), return_exp_data=True)

        if type(samples) == xr.core.dataarray.DataArray:
            samples = samples.to_dataset(name='placeholder')
            y_exp = pd.DataFrame(y_exp, columns=['placeholder'])

        y_model = samples.mean(dim='samples').to_dataframe()
        mu = y_model - y_exp
        std = samples.std(dim='samples').to_dataframe()

        self.assertAlmostEqual(mu.mean().mean(), 0, delta=self.likelihood_ordinary_obj.exp_data_formatted.max().max() * 5e-3)
        self.assertAlmostEqual(std.mean().mean(), self.sel_std_dev, delta=self.sel_std_dev*0.1)


class TestLikelihoodKineticModel(TestLikelihoodBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super(TestLikelihoodKineticModel, self).__init__(*args, **kwargs)

        self.model.create_native_odesys()
        self.sel_std_dev = 1e-7
        self.likelihood_ordinary_obj = Likelihood(self.model, OrdinaryStandardDeviation(self.sel_std_dev))


class TestLikelihoodFunction(TestLikelihoodBase):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super(TestLikelihoodFunction, self).__init__(*args, **kwargs)

        self.range = np.arange(0, 10, 0.5)

        def line(parameters):
            return self.range * parameters['m'] + parameters['b']

        def get_line_parameters():
            return {'m': 1., 'b': 5.}

        self.get_parameters = get_line_parameters
        exp_data_formatted = pd.Series(line(self.get_parameters()), index=self.range)
        self.sel_std_dev = 1e-7
        self.likelihood_ordinary_obj = Likelihood(line,
                                                  exp_data_formatted=exp_data_formatted,
                                                  std_deviation_obj=OrdinaryStandardDeviation(self.sel_std_dev))

if __name__ == '__main__':
    ut.main(verbosity=2)
