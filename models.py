def regular_autolysis():
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
