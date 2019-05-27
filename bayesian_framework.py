import matplotlib.mlab as mlab
from scipy.stats import laplace
from sampyl import np
from workflows.kinetic_modeling import default_data_format

gaussian_pdf = mlab.normpdf
laplace_pdf = laplace.pdf

class Likelihood(object):

    def __init__(self,
                 model,
                 data_conversion=default_data_format,
                 norm='gaussian',
                 std_deviation=3.3):
        self.model = model
        self.data_conversion = data_conversion
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
        try:
            exp_data = self.model.ydata_exp(data_conversion=self.data_conversion)

            # if desired, one can calculalte the maximum likelihood possible,
            # e.g. where all datapoints are perfectly modeled
            if max_likelihood:
                model_data = exp_data
            else:
                self.model.starting_concentration['poly'] = 10 ** parameters['S0']
                model_data = self.model.ydata_model_new_parameters(new_parameters=parameters,
                                                                   data_conversion=self.data_conversion)

            individual_likelihoods = []
            for idx, model_datapoint in enumerate(model_data):
                exp_datapoint = exp_data[idx]
                probability = self.calc_probability_absolute_std(exp_datapoint, model_datapoint)
                individual_likelihoods.append(probability)

            log_likelihood = np.log(individual_likelihoods).sum()
        except RuntimeError:
            log_likelihood = -np.inf

        return log_likelihood

    def calc_probability_absolute_std(self, exp_value, modeled_value):
        return self.norm(modeled_value, exp_value, self.std_deviation)

