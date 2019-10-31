import copy
import pandas as pd


def reformat_local_parameters(input_params, return_ds=False):
	local_params = copy.deepcopy(input_params)
	local_params['ka'] = local_params['kd'] + local_params['KA']

	# this corrects for the fact that we use [T-poly] (mol/L) while we should use (mol/adsorption site)
	local_params['k_LH'] = local_params['k'] - 2 * local_params['S0']
	local_params['k_ER'] = local_params['k'] - local_params['S0']

	if return_ds:
		curr_parameters_ds = pd.Series(local_params)
		return curr_parameters_ds
	else:
		return local_params
