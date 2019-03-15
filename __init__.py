from chempy import Reaction, ReactionSystem
from chempy.kinetics.ode import get_odesys
from collections import defaultdict
import copy
import inspect
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


class KineticModel(object):
    """quickly build a kinetic model fitting """

    def __init__(self, exp_data, reaction_list_input, reaction_rates=None):
        self.exp_data = exp_data
        self.studied_concentration = exp_data.columns.name

        self.reaction_list_input = reaction_list_input

        self.species = self.get_species()
        # avoid that fitted reaction rates which are fed-in are altered
        self.reaction_rates = copy.deepcopy(reaction_rates)

        self.set_binding_sites()
        self.get_starting_concentration()

        self.rate_sliders = self.create_rate_sliders()

        self.educts = self.identify_educts()
        self.products = self.identify_products()

        self.model = self.create_reaction_system()
        self.model_exp_data()

    def set_binding_sites(self):
        """keep self.binding_sites updated"""
        if 'binding_sites' in self.reaction_rates:
            self.binding_sites = self.reaction_rates['binding_sites']
        else:
            self.binding_sites = 1.0

    def identify_educts(self):
        if 'T' in self.species:
            return ['T']
        elif 'A' in self.species and 'D' in self.species:
            return ['A', 'D']
        else:
            raise ValueError('Could not identify educts of this reaction system!')

    def identify_products(self):
        product_species = []
        for name in self.species:
            # product will always be called P
            if 'P' in name:
                condition_1 = 'P0' not in name
                condition_2 = 'P$_0' not in name
                condition_3 = 'P_0' not in name

                if condition_1 and condition_2 and condition_3:
                    product_species.append(name)

        return product_species

    def get_starting_concentration(self):
        """deduce starting concentrations from the name of the DF
        if that does not work, there are no starting concentrations"""
        try:
            species_name = self.exp_data.name[0]
            species_concentration = self.exp_data.name[1]
            if 'poly' in species_name:
                species_concentration *= self.binding_sites
            starting_concentration = {species_name: species_concentration}
        except:
            starting_concentration = {}

        self.starting_concentration = starting_concentration

    @staticmethod
    def create_reaction(reaction_input, rates_dict):
        educts, products, reaction_constant_key = reaction_input
        reaction = Reaction(educts, products, 10 ** rates_dict[reaction_constant_key])
        return reaction

    def get_species(self):
        species = []

        for reactant in [0, 1]:
            for x in np.array(self.reaction_list_input)[:, reactant]:
                for entry in x.keys():
                    species.append(entry)

        unique_species = set(species)
        unique_species = list(unique_species)
        unique_species.sort()
        return unique_species

    def get_reaction_constant_names(self):
        """return list of reaction constant names, sorted in the same way as in self.reaction_rates
        this is necessary because self.reaction_rates can contain items which do not correspond to reaction rates"""
        reaction_constant_names =  [x[-1] for x in self.reaction_list_input]
        sorted_reaction_constant_names = [x for x in self.reaction_rates.index if x in reaction_constant_names]
        return sorted_reaction_constant_names

    def create_reaction_system(self):
        """create the reaction system for chempy"""
        included_reactions = [self.create_reaction(x, self.reaction_rates) for x in self.reaction_list_input]
        included_species = ' '.join(self.species)

        return ReactionSystem(included_reactions, included_species)

    def interactive_rsys(self, **kwargs):
        """interactively create a reaction system and compare modeled results to experimental data"""
        self.reaction_rates.update(pd.Series(kwargs))
        self.model = self.create_reaction_system()
        self.set_binding_sites()
        self.get_starting_concentration()
        self.model_exp_data()
        self.show_exp_data()

    def create_rate_sliders(self):
        slider_names = self.get_reaction_constant_names()

        # only create a slider if binding_sites was specified in input
        if 'binding_sites' in self.reaction_rates:
            slider_names.append('binding_sites')
        sliders = [create_rate_slider(key, self.reaction_rates) for key in slider_names]

        return pd.Series(sliders, index=slider_names)

    def interactive_plot(self):
        ipywidgets.interact(self.interactive_rsys, **self.rate_sliders)

    def evaluate_system(self, initial_concentrations, time):
        """evaluate concentration of all species at given times"""
        odesys, extra = get_odesys(self.model)

        c0 = defaultdict(float, initial_concentrations)
        result = odesys.integrate(time, c0, atol=1e-12, rtol=1e-14)

        # somehow, there's always an array full of zeros: get rid of it
        evaluation = np.array(result.at(time))[:, 0]

        # format results as DataFrame
        evaluation = pd.DataFrame(evaluation, columns=self.get_species(), index=time)
        return evaluation

    def show_exp_data(self, compare_model=True):
        if compare_model:
            ax = self.exp_data.plot(style='o', legend=False)
            # ensure same styles (esp. colors) are used for experimental and modeled data
            ax.set_prop_cycle(None)
            # iterating over columns allows to drop nan entries
            # nan entries disrupt lines
            for col in self.modeled_data:
                self.modeled_data[col].dropna().plot(ax=ax, style="--x", legend=False)

        else:
            for col in self.exp_data:
                self.exp_data[col].dropna().plot(style="-o", legend=False)

        plt.ylabel('% initial activity')
        plt.ylim(-5, 105)

    def model_exp_data(self, observable='product', only_exp_data=True, return_only=False):
        modeled_data = copy.deepcopy(self.exp_data)
        curr_starting_conc = copy.deepcopy(self.starting_concentration)

        for conc in modeled_data.columns:
            curr_starting_conc[self.studied_concentration] = conc
            if 'poly' in self.studied_concentration:
                curr_starting_conc[self.studied_concentration] *= self.binding_sites
            concentrations = self.evaluate_system(curr_starting_conc, modeled_data.index)
            educts_starting_conc = concentrations[self.educts].loc[0].sum()

            if observable == 'educt':
                # check remaining concentration of educts
                observed_activity = concentrations[self.educts].sum(axis=1)
            elif observable == 'product':
                # check how much product has been created and subtract it from starting concentration of educt
                observed_activity = concentrations[self.products].sum(axis=1)
                observed_activity = educts_starting_conc - observed_activity
            else:
                raise ValueError('Unknown observable!')
            modeled_data[conc] = observed_activity / educts_starting_conc

        # convert it to % of starting concentration
        modeled_data *= 100

        if only_exp_data:
            modeled_data = modeled_data[~np.isnan(self.exp_data)]

        if return_only:
            return modeled_data
        else:
            self.modeled_data = modeled_data

    def reaction_order_plots(self, compare_model=True):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5), sharex=True)

        axes[0].set_yscale('log')
        axes[0].set_ylabel(r'$\log_{10}$ [activity]')

        axes[1].set_ylabel('1/ rel. activity')

        if compare_model:
            (self.exp_data / 100).plot(style='o', ax=axes[0], legend=False)
            axes[0].set_prop_cycle(None)
            (self.modeled_data / 100).plot(style='--x', ax=axes[0], legend=False)

            (100 / self.exp_data).plot(style='o', ax=axes[1], legend=False)
            axes[1].set_prop_cycle(None)
            (100 / self.modeled_data).plot(style='--x', ax=axes[1], legend=False)
        else:
            (self.exp_data / 100).plot(style='-o', ax=axes[0], legend=False)
            (100 / self.exp_data).plot(style='-o', ax=axes[1], legend=False)

    def ydata_exp(self):
        """return a flattened array of the available experimental data. This is helpful for parameter fitting"""
        return self.format_ydata(self.exp_data)

    def ydata_model(self):
        """return a flattened array of the modeled data corresponding to available experimental data."""
        modeled_data_array = self.model_exp_data(only_exp_data=True, return_only=True)
        return self.format_ydata(modeled_data_array)

    @staticmethod
    def format_ydata(data_array):
        """format an array so that it can be accepted as ydata by scipy.optimize.curve_fit
        1d, no values that are np.nan"""
        flattened_array = data_array.values.flatten()
        return flattened_array[~np.isnan(flattened_array)]


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
