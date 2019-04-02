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


def apo_parameters(complete_parameters):
    """:return only parameters corresponding to apo form, excl. Ca2+"""
    if type(complete_parameters) == pd.core.frame.DataFrame:
        drop_axis = 1
    else:
        drop_axis = 0

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
