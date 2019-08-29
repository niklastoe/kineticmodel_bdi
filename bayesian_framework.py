import matplotlib.mlab as mlab
import pandas as pd
from scipy.stats import laplace
import types

import numpy as np
from workflows.kinetic_modeling import default_data_format, KineticModel

gaussian_pdf = mlab.normpdf
laplace_pdf = laplace.pdf


class Likelihood(object):

    def __init__(self,
                 model,
                 data_conversion=default_data_format,
                 exp_data=None,
                 norm='gaussian',
                 std_deviation=1.,
                 f=0.):
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
        self.f = f
        if norm == 'gaussian':
            self.norm = gaussian_pdf
            self.draw_sample = np.random.normal
        elif norm == 'laplace':
            self.norm = laplace_pdf
        else:
            raise ValueError('Unknown norm!!')
        
        # this needs a 'placeholder' because no parameters are necessary to calculate the theoretical maximum
        self.max_likelihood = self.calc_likelihood('placeholder', max_likelihood=True)
        print('Highest theoretically possible likelihood: %f' % self.max_likelihood)

    def evaluate_parameters(self, parameters, return_exp_data=False):
        """return modeled data for given parameters and model or the experimental data"""
        if return_exp_data:
            return self.model.exp_data
        else:
            if 'sigma_kin' in parameters:
                sigma_kin = 10 ** parameters['sigma_kin']
            else:
                sigma_kin = 0
            if 'f_kin' in parameters:
                f_kin = 10 ** parameters['f_kin']
            else:
                f_kin = 0
            modeled_result = self.model.model_exp_data(return_only=True,
                                                       new_parameters=parameters,
                                                       return_df=False)
            sigma = sigma_incl_factor(modeled_result, sigma_kin, f_kin)

            # get random number according to model and sigma
            samples = self.draw_sample(loc=modeled_result,
                                       scale=sigma)
            samples = pd.DataFrame(samples,
                                   columns=self.model.exp_data.columns,
                                   index=self.model.exp_data.index)

            return samples

    def calc_likelihood(self, parameters, max_likelihood=False):
        """calculate how well the model fits the data.
        If max_likelihood=True, it will return the theoretical maximum,
        i.e. if a perfect model reproduced all data points exactly."""
        if max_likelihood:
            model_data = self.exp_data
        else:
            model_data = self.generate_modeled_data(parameters)

        log_likelihood = self.log_likelihood_for_datapoints(model_data)

        return log_likelihood

    def check_model(self):
        if isinstance(self.model, KineticModel):
            return 'kinetic'
        elif isinstance(self.model, types.FunctionType):
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
                model_data = self.model.ydata_model_new_parameters(new_parameters=parameters,
                                                                   data_conversion=self.data_conversion)
            except RuntimeError:
                return np.array([np.inf] * len(self.exp_data))
        elif self.model_type == 'function':
            model_data = self.model(parameters)

        return model_data

    def log_likelihood_for_datapoints(self, model_data):
        # generate a list of likelihoods for every single data point
        individual_likelihoods = self.calc_probability_absolute_std(self.exp_data, model_data)

        # sum the logs of all likelihoods and return them
        log_likelihood = np.log(individual_likelihoods).sum()
        return log_likelihood

    def calc_probability_absolute_std(self, exp_value, modeled_value):
        sigma = sigma_incl_factor(modeled_value, self.std_deviation, self.f)
        # if any modeled_value is infinity, set sigma to 1: probability is going to be -infinity
        # and we need to avoid the error this would throw
        if np.any(np.array(modeled_value) == np.inf):
            sigma = 1.
        return self.norm(modeled_value, exp_value, sigma)


def sigma_incl_factor(modeled_value, std_deviation, f):
    variance = std_deviation ** 2 + modeled_value ** 2 * f ** 2
    sigma = np.sqrt(variance)
    return sigma
class OrdinaryStandardDeviation(object):

    def __init__(self, std_dev):
        self.std_dev = std_dev

    def return_std_dev(self, *args, **kwargs):
        return self.std_dev


class FixPlusFractionalStandardDeviation(object):

    def __init__(self, sigma_name, f_name, format='log'):
        self.sigma_name = sigma_name
        self.f_name = f_name
        self.format = format

    def extract_parameters(self, parameters):
        sigma = parameters[self.sigma_name]
        f = parameters[self.f_name]
        parameters = [sigma, f]
        if self.format == 'log':
            return np.power(10, parameters)
        elif self.format == 'regular':
            return parameters
        else:
            raise NotImplementedError('Can only deal with logarithmic nuisance parameters!')

    def calculate_std_deviation(self, sigma, f, modeled_value):
        variance = sigma ** 2 + modeled_value ** 2 * f ** 2
        std_deviation = np.sqrt(variance)
        return std_deviation

    def return_std_dev(self, modeled_value, parameters):
        sigma, f = self.extract_parameters(parameters)
        std_deviation = self.calculate_std_deviation(sigma, f, modeled_value)
        return std_deviation
