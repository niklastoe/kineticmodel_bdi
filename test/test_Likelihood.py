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

    def test_generate_D_from_theta(self):
        """Posterior prediction should allow to recover input std_dev and mu (=y_model=y_exp in this case) by using the
        correct input parameters"""

        # get 100 samples for the true parameters
        samples = []

        for x in range(100):
            samples.append(self.likelihood_ordinary_obj.generate_D_from_theta(self.get_parameters()))

        samples = xr.concat([df.to_xarray() for df in samples], "samples")

        # get the true observed values
        y_exp = self.likelihood_ordinary_obj.generate_D_from_theta(self.get_parameters(), return_exp_data=True)

        # if observed/experimental data are stored in pd.Series, need to make sure that everything ends as a DataFrame
        # otherwise, comparison will be messed up
        if type(samples) == xr.core.dataarray.DataArray:
            samples = samples.to_dataset(name='placeholder')
            y_exp = pd.DataFrame(y_exp, columns=['placeholder'])

        # compare observed data to modeled data (from parameters) and check the standard deviation
        y_model = samples.mean(dim='samples').to_dataframe()
        mu = y_model - y_exp
        std = samples.std(dim='samples').to_dataframe()

        # mu should be close to 0 (input parameters used for posterior prediction)
        self.assertAlmostEqual(mu.mean().mean(), 0, delta=self.likelihood_ordinary_obj.exp_data_formatted.max().max() * 5e-3)
        # sampled standard deviation should not differ by more than 10% from real standard deviation
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
            return self.range * 10**parameters['m'] + 10**parameters['b']

        def get_line_parameters():
            return {'m': 0., 'b': 1.}

        self.get_parameters = get_line_parameters
        exp_data_formatted = pd.Series(line(self.get_parameters()), index=self.range)
        self.sel_std_dev = 1e-7
        self.likelihood_ordinary_obj = Likelihood(line,
                                                  exp_data_formatted=exp_data_formatted,
                                                  std_deviation_obj=OrdinaryStandardDeviation(self.sel_std_dev))

if __name__ == '__main__':
    ut.main(verbosity=2)
