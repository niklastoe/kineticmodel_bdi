import pandas as pd
import xarray as xr

from workflows.usability.jupyter_compatability import agnostic_tqdm


def evaluate_parameters(parameters, model, return_exp_data=False):
    """return modeled data for given parameters and model or the experimental data"""
    if return_exp_data:
        return model.exp_data
    else:
        return model.model_exp_data(return_only=True, new_parameters=parameters, return_df=True)


def calc_confidence_intervals(parameter_df, evaluation_func, quantiles=(0.05, 0.95)):
    """return confidence intervals given parameters and evaluation function"""
    ppc_samples = []

    failed_attempts = 0
    for x in agnostic_tqdm(parameter_df.iterrows()):
        try:
            ppc_samples.append(evaluation_func(x[1]))
        except RuntimeError:
            failed_attempts += 1

    print('Failed %s times' % failed_attempts)
    ppc_samples = xr.concat([df.to_xarray() for df in ppc_samples], "samples")

    return [ppc_samples.quantile(i, dim='samples').to_dataframe().drop('quantile', axis=1) for i in quantiles]


def plot_confidence_intervals(parameter_sets, evaluation_func, sel_ax):
    confidence_lo, confidence_hi = calc_confidence_intervals(parameter_sets, evaluation_func)
    for col in confidence_lo:
        sel_ax.fill_between(confidence_lo.index, confidence_lo[col], confidence_hi[col], alpha=0.6)


def plot_lines(parameter_sets, evaluation_func, sel_ax):
    for x in agnostic_tqdm(parameter_sets.iterrows()):
        curr_results = evaluation_func(x[1])
        curr_results.plot(ax=sel_ax)
        sel_ax.set_prop_cycle(None)


def plot_posterior_predicitive_check(parameter_sets, evaluation_func, sel_ax):
    plot_confidence_intervals(parameter_sets, evaluation_func, sel_ax)
    evaluation_func('placeholder', return_exp_data=True).plot(style='o', ax=sel_ax)


def calc_iid_interval(sampler):
    acors = sampler.get_autocorr_time(tol=0)
    return int(acors.max() + 1)


def create_iid_df(sampler, reformat_parameters=None):
    iid_interval = calc_iid_interval(sampler)
    iid_points = sampler.chain[:, ::iid_interval]

    # transform to parameter df
    parameter_df = pd.DataFrame(iid_points.reshape(-1, len(sampler.parm_names)), columns=sampler.parm_names)
    if reformat_parameters is None:
        return parameter_df
    else:
        parameter_df_complete = pd.DataFrame([reformat_parameters(x[1]) for x in parameter_df.iterrows()])
        return parameter_df_complete
