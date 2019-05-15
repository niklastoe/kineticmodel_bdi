import matplotlib.mlab as mlab
from sampyl import np
from workflows.kinetic_modeling import default_data_format


def calc_probability_absolute_std(exp_value, modeled_value, abs_std=3.3):
    curr_mean = exp_value
    return mlab.normpdf(modeled_value, curr_mean, abs_std)


def calc_probability_relative_std(exp_value, modeled_value, rel_std=0.2):
    curr_mean = exp_value
    curr_std = curr_mean * rel_std
    return mlab.normpdf(modeled_value, curr_mean, curr_std)


def likelihood(parameters,
               model,
               calc_probability_func=calc_probability_absolute_std,
               data_conversion=default_data_format,
               std_deviation=3.3,
               max_likelihood=False):
    """calculate how well the model fits the data.
    If max_likelihood=True, it will return the theoretical maximum, i.e. if all data points were reproduced exactly."""
    try:
        exp_data = model.ydata_exp(data_conversion=data_conversion)

        # if desired, one can calculalte the maximum likelihood possible,
        # e.g. where all datapoints are perfectly modeled
        if max_likelihood:
            model_data = exp_data
        else:
            model_data = model.ydata_model_new_parameters(new_parameters=parameters,
                                                          data_conversion=data_conversion)

        individual_likelihoods = []
        for idx, model_datapoint in enumerate(model_data):
            exp_datapoint = exp_data[idx]
            probability = calc_probability_func(exp_datapoint, model_datapoint, std_deviation)

            individual_likelihoods.append(probability)

        log_likelihood = np.log(individual_likelihoods).sum()
    except RuntimeError:
        log_likelihood = -np.inf

    return log_likelihood
