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
                 std_deviation_obj,
                 exp_data_formatted=None,
                 data_conversion=default_data_format,
                 norm='gaussian'):
        self.model = model
        self.data_conversion = data_conversion

        self.model_type = self.check_model()
        if self.model_type == 'kinetic':
            self.exp_data_formatted = self.model.exp_data
            self.exp_data = self.generate_experimental_data()
        else:
            self.exp_data_formatted = exp_data_formatted
            self.exp_data = exp_data_formatted.values

        self.std_deviation_obj = std_deviation_obj

        if norm == 'gaussian':
            self.norm = gaussian_pdf
            self.draw_sample = np.random.normal
        elif norm == 'laplace':
            self.norm = laplace_pdf
        else:
            raise ValueError('Unknown norm!!')

        self.theta_names = find_necessary_parameters(self.calc_likelihood)

        # calculate the maxmimum likelihood possible if standard deviation is constant
        if isinstance(self.std_deviation_obj, OrdinaryStandardDeviation):
            # this needs a 'placeholder' because no parameters are necessary to calculate the theoretical maximum
            self.max_likelihood = self.calc_likelihood('placeholder', max_likelihood=True)
            print('Highest theoretically possible likelihood: %f' % self.max_likelihood)

    def evaluate_parameters(self, parameters, return_exp_data=False):
        """return modeled data for given parameters and model or the experimental data"""
        if return_exp_data:
            return self.exp_data_formatted
        else:
            if self.model_type == 'kinetic':
                modeled_result = self.model.model_exp_data(return_only=True,
                                                           new_parameters=parameters,
                                                           return_df=False)
            elif self.model_type == 'function':
                modeled_result = self.model(parameters)
            sigma = self.std_deviation_obj.return_std_dev(modeled_result, parameters)

            # get random number according to model and sigma
            samples = self.draw_sample(loc=modeled_result,
                                       scale=sigma)

            if isinstance(self.exp_data_formatted, pd.DataFrame):
                samples = pd.DataFrame(samples,
                                       index=self.exp_data_formatted.index,
                                       columns=self.exp_data_formatted.columns)
            elif isinstance(self.exp_data_formatted, pd.Series):
                samples = pd.Series(samples,
                                    index=self.exp_data_formatted.index)
            else:
                raise NotImplementedError('Need to provide formatted data as pd.DataFrame or pd.Series!')

            return samples

    def calc_likelihood(self, parameters, max_likelihood=False):
        """calculate how well the model fits the data.
        If max_likelihood=True, it will return the theoretical maximum,
        i.e. if a perfect model reproduced all data points exactly."""
        if max_likelihood:
            model_data = self.exp_data
        else:
            model_data = self.generate_modeled_data(parameters)

        sigma = self.std_deviation_obj.return_std_dev(model_data, parameters)
        log_likelihood = self.log_likelihood_for_datapoints(model_data, sigma)

        return log_likelihood

    def calc_likelihood_theta(self, theta_values):
        """calculate the likelihood if a vector with values for theta is passed"""
        parameter_dict = dict(zip(self.theta_names, theta_values))
        return self.calc_likelihood(parameter_dict)

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

    def log_likelihood_for_datapoints(self, model_data, sigma):
        # generate a list of likelihoods for every single data point
        individual_likelihoods = self.calc_probability_absolute_std(self.exp_data, model_data, sigma)

        # sum the logs of all likelihoods and return them
        log_likelihood = np.log(individual_likelihoods).sum()
        return log_likelihood

    def calc_probability_absolute_std(self, exp_value, modeled_value, sigma):
        # if any modeled_value is infinity, set sigma to 1: probability is going to be -infinity
        # and we need to avoid the error this would throw
        if np.any(np.array(modeled_value) == np.inf):
            sigma = 1.
        return self.norm(modeled_value, exp_value, sigma)


def find_necessary_parameters(function, function_kwargs={}):
    """return a list of necessary parameters for a function that takes one dictionary as input"""

    success = False
    curr_dict = {}
    # catch KeyErrors until function can be evaluated properly
    while not success:
        try:
            function(curr_dict, **function_kwargs)
            success = True
        except KeyError, e:
            missing_parm = e[0]
            curr_dict[missing_parm] = 0

    return curr_dict.keys()


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

    @staticmethod
    def calculate_std_deviation(sigma, f, modeled_value):
        variance = sigma ** 2 + modeled_value ** 2 * f ** 2
        std_deviation = np.sqrt(variance)
        return std_deviation

    def return_std_dev(self, modeled_value, parameters):
        sigma, f = self.extract_parameters(parameters)
        std_deviation = self.calculate_std_deviation(sigma, f, modeled_value)
        return std_deviation
