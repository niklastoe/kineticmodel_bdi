import numpy as np


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
