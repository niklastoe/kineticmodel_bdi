import numpy as np
import unittest as ut
import pandas as pd

from workflows.kinetic_modeling import KineticModel


class TestKineticModelBase(ut.TestCase):
    __test__ = False

    def __init__(self, *args, **kwargs):
        super(TestKineticModelBase, self).__init__(*args, **kwargs)

        self.time = np.arange(0.,3600., 300.)
        self.k = -2.0
        self.parameters = pd.Series({'k': self.k})
        self.reactions = self.generate_reactions()
        self.true_data = self.generate_true_data()
        self.model = KineticModel(self.true_data, self.reactions, self.parameters, educts='A')

    def generate_reactions(self):
        raise NotImplementedError

    def integrated_rate_law(self):
        raise NotImplementedError

    def generate_true_data(self):
        """generate a DataFrame containing kinetic data from integrated rate law"""
        c = [1e-6, 2e-6]

        results = {}
        for x in c:
            true_data = self.integrated_rate_law(x, self.time, self.k)
            true_data = pd.Series(true_data, index=self.time)
            results[x] = true_data

        true_data = pd.DataFrame(results)
        true_data.columns.name = 'A'
        return true_data

    def test_reproduction_integrated_rate_law(self):
        """kinetic model should yield identical result to integrated rate law"""
        diff = self.true_data - self.model.modeled_data
        self.assertAlmostEqual(diff.sum().sum(), 0., delta=1e-9)
        self.assertLess(diff.max().max(), 1e-9)


class TestKineticModelFirst(TestKineticModelBase):
    __test__ = True

    def generate_reactions(self):
        return [[{'A': 1}, {'P': 1}, 'k']]

    def integrated_rate_law(self, A0, t, k):
        k_applied = 10**k
        At = A0 * np.exp(-k_applied*t)
        return At


class TestKineticModelSecond(TestKineticModelBase):
    __test__ = True

    def generate_reactions(self):
        return [[{'A': 2}, {'P': 1}, 'k']]

    def integrated_rate_law(self, A0, t, k):
        k_applied = 10**k
        At = 1. / (k_applied*t + 1./A0)
        return At


if __name__ == '__main__':
    ut.main(verbosity=2)
