import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc
from tqdm import tqdm, tqdm_notebook
import xarray as xr

from kineticmodel_bdi import plot_df_w_nan


def calc_uncertainty_intervals(parameter_df, evaluation_func, quantiles=(0.16, 0.5, 0.84)):
    """return uncertainty intervals given parameters and evaluation function"""
    ppc_samples = []

    failed_attempts = 0
    for x in agnostic_tqdm(parameter_df.iterrows()):
        try:
            curr_result = evaluation_func(x[1])
            ppc_samples.append(curr_result)
        except RuntimeError:
            failed_attempts += 1

    print('Failed %s times' % failed_attempts)
    ppc_samples = xr.concat([df.to_xarray() for df in ppc_samples], "samples")

    dfs = []

    for i in quantiles:
        array = ppc_samples.quantile(i, dim='samples')
        if hasattr(array, 'name'):
            if array.name is None:
                array.name = 'placeholder'
        dfs.append(array.to_dataframe().drop('quantile', axis=1))

    return dfs


def plot_uncertainty_intervals(parameter_sets, evaluation_func, sel_ax):
    uncertainty_lo, uncertainty_median, uncertainty_hi = calc_uncertainty_intervals(parameter_sets, evaluation_func)
    plot_passed_intervals(uncertainty_lo, uncertainty_median, uncertainty_hi, sel_ax)


def plot_passed_intervals(uncertainty_lo, uncertainty_median, uncertainty_hi, sel_ax):
    for idx, col in enumerate(uncertainty_lo):
        x = list(uncertainty_lo[col].dropna().index)
        y_lo = list(uncertainty_lo[col].dropna())
        y_hi = list(uncertainty_hi[col].dropna())

        if type(sel_ax) == np.ndarray:
            curr_ax = sel_ax[idx]
        else:
            curr_ax = sel_ax
        curr_color = 'C' + str(idx)
        curr_ax.fill_between(x,
                             y_lo,
                             y_hi,
                             alpha=0.4,
                             color=curr_color)

    curr_ax.set_prop_cycle(None)
    plot_df_w_nan(uncertainty_median, ax=sel_ax)
    curr_ax.set_prop_cycle(None)
    plot_df_w_nan(uncertainty_median, ax=sel_ax)
    curr_ax.set_prop_cycle(None)


def plot_lines(parameter_sets, evaluation_func, sel_ax):
    for x in agnostic_tqdm(parameter_sets.iterrows()):
        curr_results = evaluation_func(x[1])
        plot_df_w_nan(curr_results, ax=sel_ax)
        sel_ax.set_prop_cycle(None)


def posterior_predictive_check(parameter_sets, evaluation_func, sel_ax,
                               plotting=plot_uncertainty_intervals,
                               show_model_data=True,
                               show_exp_data=True):
    if show_model_data:
        plotting(parameter_sets, evaluation_func, sel_ax)
    if show_exp_data:
        plot_df_w_nan(evaluation_func('placeholder', return_exp_data=True), ax=sel_ax, style='o')


def calc_iid_interval(sampler):

    acors = None
    # check if sampler has backend filename
    if hasattr(sampler.backend, 'filename'):
        try:
            acors = read_last_autocorrelation(sampler)
        except KeyError:
            pass

    # if autocorrelation could not be read from file, need to recalculate it
    if acors is None:
        acors = sampler.get_autocorr_time(tol=0)
    return int(acors.max() + 1)


def read_last_autocorrelation(sampler):
    return pd.read_hdf(sampler.backend.filename, key='autocorrelation').iloc[-1]


def plot_autocorrelation(sampler):
    """plot autocorrelation and plot line to see if at least 50*tau have been sampled"""
    autocorrelation_df = pd.read_hdf(sampler.backend.filename, key='autocorrelation')

    autocorrelation_df.plot()
    plt.plot(autocorrelation_df.index, autocorrelation_df.index / 50)
    plt.xlabel('steps')
    plt.ylabel('tau')


def create_iid_df(sampler, reformat_parameters=None, iid_interval=None):
    """return DataFrame that contains points separated by tau
    start at tau to discard some burn-in
    also include logp and blobs"""
    if iid_interval is None:
        iid_interval = calc_iid_interval(sampler)
    burn_in = iid_interval * 5
    iid_points = sampler.get_chain(discard=burn_in, thin=iid_interval)
    iid_lnprobability = sampler.get_log_prob(discard=burn_in, thin=iid_interval, flat=True)
    iid_blobs = sampler.get_blobs(discard=burn_in, thin=iid_interval)[:, :, 0].flatten()

    # transform to parameter df
    parameter_df = pd.DataFrame(iid_points.reshape(-1, len(sampler.parm_names)), columns=sampler.parm_names)

    blob_df = read_blob_df(iid_blobs)

    if reformat_parameters is not None:
        parameter_df = pd.DataFrame([reformat_parameters(x[1]) for x in parameter_df.iterrows()])

    parameter_df['logp'] = iid_lnprobability
    # merge both DataFrames
    full_df = parameter_df.T.append(blob_df.T).T
    return full_df


def read_blob_df(blob_list):
    """return DataFrame for blobs passed as a flat list"""
    all_dicts = []
    for x in blob_list:
        # sometimes, the closing bracket is missing
        # maybe even more is lost but since those are decimals, we do not care much
        if x[-1] != '}':
            x += '}'

        # sometimes, reading in the blobs will fail nevertheless because the string in blobs is cutoff
        # in these cases, just set everything to NaN and discard the points
        # we have enough points, as long as we do not expect that points in a certain regime cause this error...
        # ...more frequently, there's no bias introduced
        try:
            curr_dict = json.loads(x)
        except ValueError:
            print('Failure: ' + x)
            curr_dict = {x: np.nan for x in curr_dict.keys()}
        all_dicts.append(curr_dict)
    blob_df = pd.DataFrame(all_dicts)
    return blob_df


def calc_gelman_rubin(sampler_a, sampler_b):
    """calculate R_hat according to Gelman-Rubin for two chains"""
    # identify longer chain
    lengths = [x.shape[0] for x in [sampler_a, sampler_b]]
    min_length = min(lengths)

    # both chains need to have identical length
    trace_a = sampler_a[:min_length]
    trace_b = sampler_b[:min_length]

    trace_array = np.array([trace_a, trace_b])

    return pymc.diagnostics.gelman_rubin(trace_array)


def gelman_rubin_emcee_samplers(sampler_a, sampler_b):
    chain_a = sampler_a.parm_df[sampler_a.parm_names].values
    chain_b = sampler_b.parm_df[sampler_a.parm_names].values

    R_hat = calc_gelman_rubin(chain_a, chain_b)

    return pd.Series(R_hat, index=sampler_a.parm_names)


def calc_reaction_rate(kinetic_data_df):
    """return reaction rate in M/s for given dataframe of kinetic data"""
    reaction_rates = []

    for x in kinetic_data_df.iterrows():
        for y in kinetic_data_df.iterrows():
            t0, t1 = x[0], y[0]
            t_diff = t1 - t0
            if t_diff > 0:
                c_diff = x[1] - y[1]
                reaction_rates.append(c_diff / t_diff)

    reaction_rates = np.array(reaction_rates).flatten()
    # get rid of nan entries
    reaction_rates = reaction_rates[~np.isnan(reaction_rates)]
    log_reaction_rates = np.log10(reaction_rates)

    return log_reaction_rates.max()


def control_factor(model, curr_parameters, sel_parm, reformat_parameters=None):
    """determine control factor for a given parameter to see if is rate determining or not
    definition of control factors: taken from Parmon, Reaction Kinetics and Catalysis Letters, 2003, 79, 303-317"""
    myparms = copy.deepcopy(curr_parameters)
    if reformat_parameters is not None:
        myparms = reformat_parameters(myparms)
    kin_data = model.model_exp_data(return_only=True, return_df=True,
                                    new_parameters=myparms)
    org_rate = calc_reaction_rate(kin_data)

    change_interval = -0.1
    myparms[sel_parm] += change_interval
    if reformat_parameters is not None:
        myparms = reformat_parameters(myparms)
    kin_data = model.model_exp_data(return_only=True, return_df=True, new_parameters=myparms)
    new_rate = calc_reaction_rate(kin_data)

    CF_abs = (new_rate - org_rate) / change_interval

    return CF_abs


def agnostic_tqdm(*args, **kwargs):
    """chooses which tqdm version to use and passes all arguments
    tqdm can be turned off (disable=True), useful if there are many loops etc"""

    def is_notebook():
        """check if code is executed in jupyter notebook or not"""
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    if is_notebook():
        sel_func = tqdm_notebook    # only works within notebook
    else:
        sel_func = tqdm  # only works outside notebook

    return sel_func(*args,  **kwargs)   # all args and kwargs are preserved
