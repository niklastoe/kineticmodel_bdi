import pandas as pd


def regular_autolysis_MM():
    """regular autolysis, matching Michaelis-Menten"""
    reactions = []

    # regular autolysis
    reactions.append([{'T': 2}, {r'T$_2$': 1}, 'TT_association'])
    reactions.append([{r'T$_2$': 1}, {'T': 2}, 'TT_dissociation'])
    reactions.append([{r'T$_2$': 1}, {'TP': 1}, 'digest_speed'])
    reactions.append([{'TP': 1}, {'T': 1, 'P': 1}, 'TP_dissociation'])

    return reactions


def regular_autolysis_Nord1956():
    """identical to regular autolysis but trypsin can be found in active (A) and denatured (B)
    conformation, they can convert back and forth"""

    reactions = []

    reactions.append([{'A': 1}, {'D': 1}, 'k_AD'])
    reactions.append([{'D': 1}, {'A': 1}, 'k_DA'])

    reactions.append([{'A': 1, 'D': 1}, {'AD': 1}, 'TT_association'])
    reactions.append([{'AD': 1}, {'A': 1, 'D': 1}, 'TT_dissociation'])
    reactions.append([{'AD': 1}, {'AP': 1}, 'digest_speed'])
    reactions.append([{'AP': 1}, {'A': 1, 'P': 1}, 'TP_dissociation'])

    return reactions


def regular_autolysis_minimal():
    return [[{'T': 2}, {'T': 1, 'P': 1}, 'k_reg']]


def complex_formation_equilibrium():
    return [[{'T': 1, 'S': 1}, {'T-S': 1}, 'ka'],
            [{'T-S': 1}, {'T': 1, 'S': 1}, 'kd']]


def langmuir_hinshelwood():
    return [[{'T-S': 2}, {'T-S': 1, 'P': 1, 'S': 1}, 'k_LH']]


def eley_rideal():
    return [[{'T': 1, 'T-S': 1}, {'T': 1, 'S': 1, 'P': 1}, 'k_ER']]


start_rates_ER = pd.Series({'KA': 5.700000,
                            'S0': -6.568636,
                            'k': -5.468636,
                            'k_reg': 1.100000,
                            'kd': -1.000000,
                            'ka': 4.700000,
                            'k_LH': 7.668636,
                            'k_ER': 1.100000})


start_rates_LH = pd.Series({'KA': 5.700000,
                            'S0': -6.568636,
                            'k': -12.037272,
                            'k_reg': 1.100000,
                            'kd': -1.000000,
                            'ka': 4.700000,
                            'k_LH': 1.100000,
                            'k_ER': -5.468636})
