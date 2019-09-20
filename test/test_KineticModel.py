import numpy as np
import unittest as ut
import pandas as pd

from workflows.kinetic_modeling import KineticModel

class CompareKineticResults(ut.TestCase):

    def compare_two_kinetic_results(self, resultA, resultB):
        diff = resultA - resultB
        self.assertAlmostEqual(diff.sum().sum(), 0., delta=1e-11)
        self.assertLess(diff.max().max(), 1e-11)
        self.assertLess(abs(diff.min().min()), 1e-11)

class TestKineticModelBase(CompareKineticResults):
    __test__ = False

    def __init__(self, *args, **kwargs):
        super(TestKineticModelBase, self).__init__(*args, **kwargs)

        self.time = np.arange(0., 3600., 300.)
        self.parameters = self.get_parameters()
        self.reactions = self.generate_reactions()
        self.true_data = self.generate_true_data()
        self.model = KineticModel(self.true_data, self.reactions, self.parameters, educts='A')
        self.model.create_native_odesys()

    def get_parameters(self):
        raise NotImplementedError

    def generate_reactions(self):
        raise NotImplementedError

    def integrated_rate_law(self):
        raise NotImplementedError

    def generate_true_data(self):
        """generate a DataFrame containing kinetic data from integrated rate law"""
        c = [1e-6, 2e-6]

        results = {}
        for x in c:
            true_data = self.integrated_rate_law(x, self.time, self.parameters['k'])
            true_data = pd.Series(true_data, index=self.time)
            results[x] = true_data

        true_data = pd.DataFrame(results)
        true_data.columns.name = 'A'
        return true_data

    def test_reproduction_integrated_rate_law(self):
        """kinetic model should yield identical result to integrated rate law"""
        self.compare_two_kinetic_results(self.true_data, self.model.modeled_data)

    def test_species(self):
        self.assertEqual(self.model.species, ['A', 'P'])
        self.assertEqual(self.model.starting_concentration, {'A': 0.0, 'P': 0.0})

    def test_native_odesys(self):
        """check that native odesys yields the same result as symbolic odesys"""
        c0 = {'A': 1e-6}
        org_results = self.model.evaluate_system(c0)

        native_results = self.model.evaluate_system(c0, self.model.parameters)
        self.compare_two_kinetic_results(org_results, native_results)

    def test_observables(self):
        """monitoring educt or product needs to yield the same result"""
        educt = self.model.model_exp_data(observable='educt', return_only=True)
        product = self.model.model_exp_data(observable='product', return_only=True)
        self.compare_two_kinetic_results(educt, product)

    # this should go last because it manipulates self.model
    def test_starting_conc_as_parameter(self):
        new_parameters = pd.Series({'k': self.parameters['k'], 'P0': -7})
        self.model.species_w_variable_starting_concentration = ['P0']
        new_c0 = self.model.update_starting_concentration(new_parameters)
        self.assertEqual(new_c0['P'], 1e-7)


class TestKineticModelFirst(TestKineticModelBase):
    __test__ = False

    def get_parameters(self):
        return pd.Series({'k': -2.})

    def generate_reactions(self):
        return [[{'A': 1}, {'P': 1}, 'k']]

    def integrated_rate_law(self, A0, t, k):
        k_applied = 10**k
        At = A0 * np.exp(-k_applied*t)
        return At


class TestKineticModelSecond(TestKineticModelBase):
    __test__ = True

    def get_parameters(self):
        return pd.Series({'k': 3.})

    def generate_reactions(self):
        return [[{'A': 2}, {'P': 1, 'A': 1}, 'k']]

    def integrated_rate_law(self, A0, t, k):
        k_applied = 10**k
        At = 1. / (k_applied*t + 1./A0)
        return At


class TestKineticModelEquilibrium(CompareKineticResults):
    __test__ = True

    def __init__(self, *args, **kwargs):
        super(TestKineticModelEquilibrium, self).__init__(*args, **kwargs)

        times = np.arange(0, 3600, 60)
        conc_A = np.array([10e-6] * len(times))
        exp_data = pd.DataFrame({10e-6: conc_A, 50e-6: 5*conc_A}, index=times)
        exp_data.columns.name = ['A', 'D']

        reactions = [[{'A': 1}, {'D': 1}, 'k_AD'],
                     [{'D': 1}, {'A': 1}, 'k_DA']]

        parameters = pd.Series({'k_AD': 1., 'k_DA': 1.})

        self.model = KineticModel(exp_data, reactions, parameters)

    def test_equilibrium_works(self):
        # if we look at the sum of A and D, the exp_data is correct...
        self.compare_two_kinetic_results(self.model.exp_data,
                                         self.model.model_exp_data(observable=['A', 'D'], return_only=True))
        # ...the concentration of A alone should be half as much
        self.compare_two_kinetic_results(self.model.exp_data * 0.5,
                                         self.model.model_exp_data(observable='A', return_only=True))


if __name__ == '__main__':
    ut.main(verbosity=2)
