import matplotlib.mlab as mlab
from scipy.stats import laplace
import types

from sampyl import np
from workflows.kinetic_modeling import default_data_format, KineticModel

gaussian_pdf = mlab.normpdf
laplace_pdf = laplace.pdf

class Likelihood(object):

    def __init__(self,
                 model,
                 data_conversion=default_data_format,
                 exp_data=None,
                 norm='gaussian',
                 std_deviation=3.3):
        self.model = model
        self.data_conversion = data_conversion

        self.model_type = self.check_model()

        if self.model_type == 'kinetic':
            self.exp_data = self.generate_experimental_data()
        else:
            if exp_data is None:
                raise ValueError('You need to specify the experimental data!')
            else:
                self.exp_data = exp_data

        self.std_deviation = std_deviation
        if norm == 'gaussian':
            self.norm = gaussian_pdf
        elif norm == 'laplace':
            self.norm = laplace_pdf
        else:
            raise ValueError('Unknown norm!!')
        self.max_likelihood = self.calc_likelihood('placeholder', max_likelihood=True)
        print('Highest theoretically possible likelihood: %f' % self.max_likelihood)

    def calc_likelihood(self, parameters, max_likelihood=False):
        """calculate how well the model fits the data.
        If max_likelihood=True, it will return the theoretical maximum, i.e. if all data points were reproduced exactly."""
        if max_likelihood:
            model_data = self.exp_data
        else:
            model_data = self.generate_modeled_data(parameters)

        log_likelihood = self.log_likelihood_for_datapoints(model_data)

        return log_likelihood

    def check_model(self):
        if type(self.model) == KineticModel:
            return 'kinetic'
        elif type(self.model) == types.FunctionType:
            return 'function'
        else:
            raise ValueError('Model not understood!!')


    def generate_experimental_data(self):
        return self.model.ydata_exp(data_conversion=self.data_conversion)

    def generate_modeled_data(self, parameters):
        if self.model_type == 'kinetic':
            # catch possible RuntimeErrors: set value to infinity,
            # this will give likelihood of zero, log likelihood of -infinity
            try:
                self.model.starting_concentration['poly'] = 10 ** parameters['S0']
                model_data = self.model.ydata_model_new_parameters(new_parameters=parameters,
                                                                   data_conversion=self.data_conversion)
            except RuntimeError:
                return np.array([np.inf] * len(self.exp_data))
        elif self.model_type == 'function':
            model_data = self.model(parameters)

        return model_data

    def log_likelihood_for_datapoints(self, model_data):
        individual_likelihoods = []
        for idx, model_datapoint in enumerate(model_data):
            exp_datapoint = self.exp_data[idx]
            probability = self.calc_probability_absolute_std(exp_datapoint, model_datapoint)
            individual_likelihoods.append(probability)
        log_likelihood = np.log(individual_likelihoods).sum()
        return log_likelihood

    def calc_probability_absolute_std(self, exp_value, modeled_value):
        return self.norm(modeled_value, exp_value, self.std_deviation)

