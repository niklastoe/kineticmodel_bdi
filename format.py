import numpy as np
import pandas as pd

from workflows.data_structure_manipulation import alter_listoflists_hierarchy


def michaelis_constant(parameters):
    """calculate Michaelis constant K_M"""
    return np.log10(10**parameters['TT_dissociation'] + 10**parameters['digest_speed']) - parameters['TT_association']


def kad_equilibrium(parameters):
    """calculate equilibrium constant for A <-> D equilibrium
    K_AD = k_{D->A} / k_{A->D}"""
    return parameters['DA'] - parameters['AD']


def catalytic_efficiency(parameters):
    """calculat k_cat / K_M, catalytic efficiency"""
    km = michaelis_constant(parameters)

    return parameters['digest_speed'] - km


def identify_holo(parameters, return_counterpart=False):
    """identify parameters that are related to Ca2+ bound (holo) form"""

    holo_identifier = '_CA'

    if type(parameters) == pd.core.frame.DataFrame:
        parameter_entries = parameters.columns
    else:
        parameter_entries = parameters.index
    holo_entries = [x for x in parameter_entries if holo_identifier in x]

    if not return_counterpart:
        return holo_entries
    else:
        counterparts = [x.replace(holo_identifier, '') for x in holo_entries]
        apo_holo_pairs = alter_listoflists_hierarchy([counterparts, holo_entries])
        return apo_holo_pairs


def identify_axis(parameters):
    """:return the axis where parameter names are stored"""
    if type(parameters) == pd.core.frame.DataFrame:
        drop_axis = 1
    else:
        drop_axis = 0
    return drop_axis


def apo_parameters(complete_parameters):
    """:return only parameters corresponding to apo form, excl. Ca2+"""
    drop_axis = identify_axis(complete_parameters)

    holo_entries = identify_holo(complete_parameters)
    apo_parameters_ds = complete_parameters.drop(holo_entries, axis=drop_axis)
    return apo_parameters_ds


def holo_parameters(complete_parameters):
    """:return only parameters corresponding to holo form, incl. Ca2+"""
    holo_parameters_ds = apo_parameters(complete_parameters)
    apo_holo_entries = identify_holo(complete_parameters, return_counterpart=True)

    for sel_entry in apo_holo_entries:
        holo_parameters_ds[sel_entry[0]] = complete_parameters[sel_entry[1]]
    return holo_parameters_ds


def rename_parameters(parameters):
    """rename all parameters with fitting LaTeX forms"""
    name_axis = identify_axis(parameters)
    texed_names = {'KM': r'$K_M$',
                   'posterior': r'$p(\theta | D)$',
                   'prior': r'$p(D)$',
                   'likelihood': r'$p(D | \theta)$',
                   'KAD': r'$K_{AD}$',
                   'KKM': r'$k_{cat}$ / $K_M$',
                   'AD': r'$k_{AD}$',
                   'DA': r'$k_{DA}$',
                   'TT_association': r'$k_{f}$',
                   'TT_dissociation': r'$k_{r}$',
                   'TP_dissociation': r'$k_{TP}$',
                   'digest_speed': r'$k_{cat}$',
                   'dPdt': r'$\frac{dP}{dt}$'}

    for x in parameters.axes[name_axis]:
        for y in texed_names.keys():
            if y == x[:len(y)] and y != x:
                texed_names[x] = texed_names[y] + x.replace(y, '')

    return parameters.rename(texed_names, axis=name_axis)
