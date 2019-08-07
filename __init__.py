from chempy import Reaction, ReactionSystem
from chempy.kinetics.ode import get_odesys
from pyodesys.native import native_sys
from collections import defaultdict
import copy
import inspect
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

default_data_format = 'absolute'


class KineticModel(object):
    """quickly build a kinetic model fitting """

    def __init__(self, exp_data, reaction_list_input,
                 parameters=None, observed_species='product', educts=None):
        self.exp_data = exp_data
        # check if there is information on the starting concentrations, otherwise use empty dictionary
        try:
            self.starting_concentration = exp_data.starting_concentration
        except:
            self.starting_concentration = {}
        self.studied_concentration = exp_data.columns.name
        self.observed_species = observed_species

        self.reaction_list_input = reaction_list_input

        self.species = self.identify_species()
        # avoid that input reaction rates are altered, e.g. by self.interactive_plot
        self.parameters = copy.deepcopy(parameters)

        if educts is None:
            self.educts = self.identify_educts()
        else:
            self.educts = educts
        self.products = self.identify_products()
        self.species_w_variable_starting_concentration = self.variable_starting_concentrations()

        self.data_conversion_dict = self.data_conversion_options()

        self.setup_model()
        self.model_exp_data()

    def update_starting_concentration(self, parameters=None):
        """update a starting concentration specified as a parameter"""

        if parameters is None:
            parameters = self.parameters

        curr_starting_concentration = copy.deepcopy(self.starting_concentration)

        for x in self.species_w_variable_starting_concentration:
            species_name = x[:-1]
            curr_starting_concentration[species_name] = 10**parameters[x]

        return curr_starting_concentration

    def variable_starting_concentrations(self):
        """identify species with variable starting concentrations (i.e. those where the starting concentration is
        controlled by a parameter of the model"""
        variable_species = []
        for x in self.parameters.index:
            if x[-1] == '0':
                variable_species.append(x)
        return variable_species

    def setup_model(self):
        """perform all necessary steps to setup the model and make it ready to reproduce the experimental data"""
        self.starting_concentration = self.update_starting_concentration()
        self.model = self.create_reaction_system()
        # create system of ordinary differential equations
        self.odesys, extra = get_odesys(self.model, include_params=False)

    def data_conversion_options(self):
        """return a dictionary of options how to convert/present the kinetic data"""
        options = {}

        # absolute concentration (this is the default/data must be read in as absolute concentrations)
        def absolute(x):
            return x
        options['absolute'] = {'func': absolute,
                               'ylabel': '[T] / M'}

        # log10 of absolute concentration
        def log10_absolute(x):
            return np.log10(absolute(x))
        options['log10_absolute'] = {'func': log10_absolute,
                                     'ylabel': r'$\log_{10}$ [T] / M'}

        # initial_activity (remaining percentage of initial activity, dropping from 100 to 0%)
        def initial_activity(x):
            return np.divide(x * 100, self.exp_data.columns.values)
        options['initial_activity'] = {'func': initial_activity,
                                       'ylabel': '% initial activity',
                                       'ylim': (-5, 105)}

        # log10 of initial activity
        def log10_initial_activity(x):
            return np.log10(initial_activity(x))
        options['log10'] = {'func': log10_initial_activity,
                            'ylabel': r'$\log_{10}$ activity / %'}

        # invert of initial activity (if linear this is indicative of a second-order reaction)
        def invert(x):
            return 100. / initial_activity(x)
        options['invert'] = {'func': invert,
                             'ylabel': '1 / rel. activity'}

        return options

    def identify_educts(self):
        """find all species which are considered educts. This is based on my trypsin models"""
        if 'T' in self.species:
            return ['T']
        elif 'A' in self.species and 'D' in self.species:
            return ['A', 'D']
        else:
            raise ValueError('Could not identify educts of this reaction system!')

    def identify_products(self):
        """find all species which are considered educts.
        This function assumes that all of those species contain a capital P"""
        product_species = []
        for name in self.species:
            # product will always be called P
            if 'P' in name:
                # sometimes, there will be species which can carry numerous P, check that it's not 0
                condition_1 = 'P0' not in name
                condition_2 = 'P$_0' not in name
                condition_3 = 'P_0' not in name

                if condition_1 and condition_2 and condition_3:
                    product_species.append(name)

        return product_species

    def identify_species(self):
        species = []

        for reactant in [0, 1]:
            for x in np.array(self.reaction_list_input)[:, reactant]:
                for entry in x.keys():
                    species.append(entry)

        unique_species = set(species)
        unique_species = list(unique_species)
        unique_species.sort()
        return unique_species

    def get_slider_names(self):
        """return list of slider names, sorted in the same way as in self.parameters
        this is necessary because self.parameters can contain items which won't be sliders"""
        slider_names = [x[-1] for x in self.reaction_list_input]

        # create sliders for variable starting conditions
        for species_parameter in self.species_w_variable_starting_concentration:
            slider_names.append(species_parameter)

        sorted_slider_names = [x for x in self.parameters.index if x in slider_names]
        return sorted_slider_names

    def create_reaction_system(self):
        """create the reaction system for chempy"""
        included_reactions = [Reaction(*x) for x in self.reaction_list_input]
        included_species = ' '.join(self.species)

        return ReactionSystem(included_reactions, included_species)

    def interactive_rsys(self, **kwargs):
        """interactively create a reaction system and compare modeled results to experimental data"""
        self.parameters.update(pd.Series(kwargs))
        self.setup_model()
        self.model_exp_data()
        self.show_exp_data(data_conversion=kwargs['format'])

    def create_rate_sliders(self):
        slider_names = self.get_slider_names()

        sliders = [create_rate_slider(key, self.parameters) for key in slider_names]

        # add selection for data conversion
        data_conversion_slider = ipywidgets.RadioButtons(options=self.data_conversion_dict.keys(),
                                                         value=default_data_format,
                                                         description='format')
        sliders.append(data_conversion_slider)
        slider_names += ['format']

        return pd.Series(sliders, index=slider_names)

    def interactive_plot(self):
        rate_sliders = self.create_rate_sliders()
        ipywidgets.interact(self.interactive_rsys, **rate_sliders)

    def evaluate_system(self, initial_concentrations, time, new_parameters=None, return_df=False):
        """evaluate concentration of all species at given times"""
        c0 = defaultdict(float, initial_concentrations)

        # add early integration steps, this helps somehow with stability
        early_steps = [time[1] / 10**i for i in np.arange(1,5)] # it should be very early indeed
        integration_times = list(time) + early_steps
        integration_times.sort()
        # we will need the indeces of the original times later!
        org_time_index = [integration_times.index(x) for x in time]
        integration_times = np.array(integration_times)

        def convert_parameters(input_parameters):
            """convert pd.Series to dict where all rates are 10**x
            WARNING: you cannot pass the pd.Series directly,
            it will be treated like my_ds.values and rates will be wrongly associated!!"""

            input_type = type(input_parameters)
            if input_type == pd.core.series.Series:
                return convert_parameters(input_parameters.to_dict())
            elif input_type == dict:
                formatted_parameters = {}
                for x in input_parameters:
                    formatted_parameters[x] = 10 ** input_parameters[x]
                return formatted_parameters
            else:
                raise ValueError('New parameters must either be pd.Series or dictionary!!')

        if new_parameters is None:
            result = self.odesys.integrate(integration_times, c0, convert_parameters(self.parameters),
                                           atol=1e-12, rtol=1e-14)
        else:
            # native odesys requires that all species have a starting concentration
            for x in self.species:
                if x not in c0:
                    c0[x] = 0.0
            result = self.native_odesys.integrate(integration_times, c0, convert_parameters(new_parameters))

        # just get the concentrations at the input time steps; drop early_steps
        evaluation = result.yout[org_time_index]

        # by default, avoid pandas to save time. If desired, it is possible to format everything nicely in a DataFrame
        if return_df:
            return pd.DataFrame(evaluation, columns=self.species, index=time)
        else:
            return evaluation

    def show_exp_data(self, compare_model=True, legend=False, data_conversion=default_data_format):
        """show a plot of the experimental data. By default, modeled data will be shown as well"""
        selected_conversion = self.data_conversion_dict[data_conversion]
        data_conversion_func = selected_conversion['func']

        exp_data = data_conversion_func(self.exp_data)
        modeled_data = data_conversion_func(self.modeled_data)

        if compare_model:
            ax = exp_data.plot(style='o', legend=False)
            # ensure same styles (esp. colors) are used for experimental and modeled data
            ax.set_prop_cycle(None)
            plot_df_w_nan(modeled_data, style="--x", axes=ax)

        else:
            plot_df_w_nan(exp_data, style="-o")

        plt.ylabel(selected_conversion['ylabel'])
        if 'ylim' in selected_conversion:
            plt.ylim(*selected_conversion['ylim'])
        if legend:
            plt.legend(self.exp_data.columns)

    def model_exp_data(self,
                       observable='default',
                       only_exp_data=True,
                       return_only=False,
                       new_parameters=None):
        """model the experimental data and return a pd.DataFrame which matches the format of the experimental one.
        If new parameters are given, this will use the native (C++) odesys and avoid pandas for speed up!"""
        if observable == 'default':
            observable = self.observed_species

        # get starting concentrations and update them if needed
        if new_parameters is None:
            curr_starting_conc = copy.deepcopy(self.starting_concentration)
        else:
            curr_starting_conc = self.update_starting_concentration(new_parameters)
        modeled_data = []

        for conc_idx, conc in enumerate(self.exp_data.columns):
            curr_starting_conc[self.studied_concentration] = conc
            concentrations = self.evaluate_system(curr_starting_conc,
                                                  self.exp_data.index,
                                                  new_parameters=new_parameters)
            educts_starting_conc = [curr_starting_conc[x] for x in self.educts if x in curr_starting_conc]

            if observable == 'educt':
                # check remaining concentration of educts
                observed_activity = self.get_species_concentration(concentrations, self.educts)
            elif observable == 'product':
                # check how much product has been created and subtract it from starting concentration of educt
                observed_activity = self.get_species_concentration(concentrations, self.products)
                observed_activity = educts_starting_conc - observed_activity
            elif observable in self.species:
                """check concentration of any species of interest"""
                observed_activity = self.get_species_concentration(concentrations, [observable])
            elif type(observable) == list:
                """check concentration of any species of interest"""
                observed_activity = self.get_species_concentration(concentrations, observable)
            else:
                raise ValueError('Unknown observable!')
            modeled_data.append(observed_activity)

        modeled_data = np.array(modeled_data).T

        if only_exp_data:
            modeled_data[np.where(np.isnan(self.exp_data))] = np.nan

        # only format as pd.DataFrame if speed is not necessary
        if new_parameters is None:
            modeled_data = pd.DataFrame(modeled_data,
                                        index=self.exp_data.index,
                                        columns=self.exp_data.columns)

        if return_only:
            return modeled_data
        else:
            self.modeled_data = modeled_data

    def get_species_concentration(self, concentrations, observables):
        species_indeces = [self.species.index(x) for x in observables]
        observed_activity = concentrations[:, species_indeces].sum(axis=1)
        return observed_activity

    def reaction_order_plots(self, compare_model=True):
        """log10 plot of initial activity should be linear for first-order reaction,
        inverted initial activity should be linear for second-order reaction"""
        for plot_style in ['log10', 'invert']:
            self.show_exp_data(compare_model=compare_model, data_conversion=plot_style)

    def ydata_exp(self, data_conversion=default_data_format):
        """return a flattened array of the available experimental data. This is helpful for parameter fitting"""
        return self.format_ydata(self.exp_data, data_conversion=data_conversion)

    def ydata_model(self, data_conversion=default_data_format):
        """return a flattened array of the modeled data corresponding to available experimental data."""
        return self.format_ydata(self.modeled_data, data_conversion=data_conversion)

    def ydata_model_new_parameters(self, new_parameters, data_conversion=default_data_format):
        new_modeled_data = self.model_exp_data(new_parameters=new_parameters, return_only=True)

        return self.format_ydata(new_modeled_data, data_conversion=data_conversion)

    def format_ydata(self, data_array, data_conversion=default_data_format):
        """format an array so that it can be accepted as ydata by scipy.optimize.curve_fit
        1d, no values that are np.nan
        By default, it will return data as % of initial activity, it can use all data_conversions, though!"""
        if type(data_array) == pd.core.frame.DataFrame:
            organized_array = data_array.values
        else:
            organized_array = data_array

        # transform data if desired
        organized_array = self.data_conversion_dict[data_conversion]['func'](organized_array)

        flattened_array = organized_array.flatten()

        return flattened_array[~np.isnan(flattened_array)]

    def create_native_odesys(self):
        """create a system of the ordinary differential equations in native C++ code.
        It is much faster than the other one for numerous sets of parameters but requires some time for setup."""
        self.native_odesys = native_sys['cvode'].from_other(self.odesys)


def plot_df_w_nan(df, style, axes=None, alpha=1.):
    """iterating over columns allows to drop nan entries
    nan entries disrupt lines"""

    # create axes if necessary
    if axes is None:
        axes = plt.axes()
    for col in df:
        df[col].dropna().plot(style=style, ax=axes, legend=False, alpha=alpha)


def create_rate_slider(rate_key, rates_dict=None, slider_range=5):
    """return a continous slider named rate_key with a given start value"""
    if rates_dict is None:
        start_val = 0
    else:
        start_val = rates_dict[rate_key]
    return ipywidgets.FloatSlider(value=start_val,
                                  min=start_val - slider_range,
                                  max=start_val + slider_range,
                                  step=0.1,
                                  description=rate_key,
                                  disabled=False,
                                  continuous_update=False,
                                  orientation='horizontal',
                                  readout=True,
                                  readout_format='.1f')


def optimize_kinetic_model(func_to_optimize, experimental_data, bounds=5):

    # prepare names and starting values for the parameters
    reaction_rates_guess = get_rates_dict_guess(func_to_optimize)
    reaction_rates_names = reaction_rates_guess.index
    starting_values_rates = reaction_rates_guess.values

    # format experimental ydata (the ones the optimizer shall match) by building
    ydata = KineticModel.format_ydata(experimental_data)

    if type(bounds) == int or type(bounds) == float:
        upper_bounds = starting_values_rates + bounds
        lower_bounds = starting_values_rates - bounds
    if type(bounds) == tuple:
        lower_bounds, upper_bounds = bounds

    if bounds is None:
        popt, pcov = curve_fit(func_to_optimize, 'placeholder', ydata)
    else:
        popt, pcov = curve_fit(func_to_optimize, 'placeholder', ydata, bounds=(lower_bounds, upper_bounds))

        # format reaction rates nicely in a pd.Series before returning it
    popt_ds = pd.Series(popt, index=reaction_rates_names)
    return popt_ds


def retrieve_ydata(experimental_data, reactions_list, reaction_rates):
    """model experimental data and format it like curve_fit needs it
    this only requires adapting the input of the experimental data"""
    reaction_rates_ds = pd.Series(reaction_rates)
    parametrized_model = KineticModel(experimental_data, reactions_list, reaction_rates_ds)

    ydata = parametrized_model.ydata_model()
    return ydata


def get_rates_dict_guess(func_to_inspect):
    """create reaction rate dictionary from keywords of function"""
    func_inspection = inspect.getargspec(func_to_inspect)
    rates_ds = pd.Series(func_inspection.defaults, index=func_inspection.args[1:])
    return rates_ds


def poly_string(x, y):
    """generate species string for polymer with"""
    return r'T$_%d$P$_%d$-poly' % (x, y)
