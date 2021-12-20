"""
    Access to the data description of the turtle cortical circuit obtained by Mike et al.
"""
import logging
import pandas as pd

from tctx.util import networks, track

# Baseline firing rates in turtle dorsal cortex are (median, 25, 75 percentile)
# - *ex vivo*: E: 0.01 [0, 0.05] spks/s I: 0.01 [0, 0.04] (Hemberger et al. 2019)
# - *in vivo*: 0.03 [0.02, 0.09] spks/s (Fournier et al. 2018)
#
# Carbachol experiments (Hemberger et al. 2019):
# - control: 0.02 +- 0.05
# - carbachol: 0.1 +- 0.25

DEFAULT_ACT_BINS = {
    'low_act': (.0, .05),
    'high_act': (.02, .09),
}


def get_layers_desc() -> pd.DataFrame:
    filename = track.ExperimentTracker().get_external_data_path('layers_desc.csv')
    logging.debug('loading %s', filename)
    return pd.read_csv(filename, index_col=0)


def get_ei_density():
    layers_desc = get_layers_desc()

    densities = pd.Series({
        'e': layers_desc.loc['L2', 'density_per_mm2'],
        'i': (layers_desc.loc['L1', 'density_per_mm2'] + layers_desc.loc['L3', 'density_per_mm2']),
    })

    densities.name = 'density_per_mm2'

    return densities


def get_connectivity_profiles():
    # TODO these should be read from a file written in the analysis step
    return {
        'e2e_std': 200,
        'e2i_std': 200,
        'i2e_std': 200,
        'i2i_std': 200,
    }


def get_reversal_potentials():
    return {
        'E_ex': +10.0,  # Excitatory reversal potential (mV) (default: 0.0)
        'E_in': -75.0,  # Inhibitory reversal potential (mV) (default: -85.0)
        'v_clamp_ex': -70.,
        'v_clamp_in': -50.,
        'ei_current_ratio': 2.5,
    }


def get_base_params_neuron():
    # The following params come from Nest's defaults for aeif_cond_exp
    # see: http://www.nest-simulator.org/cc/aeif_cond_exp/
    # see: https://doi.org/10.1152/jn.00686.2005

    default_params = {
        # Dynamic state variables:
        # 'V_m':  , # Membrane potential (mV)
        # 'g_ex': , # Excitatory synaptic conductance (nS)
        # 'g_in': ,  #  Inhibitory synaptic conductance in nS.
        # 'w': ,  #  Spike-adaptation current in pA.

        # Membrane Parameters:
        't_ref': 2.0,  # Duration of refractory period (ms) (default: 0.0)
        'V_reset': -60.0,  # Reset value for V_m after a spike (mV) (default: -60.0)
        'E_L': -70.6,  # Leak reversal potential (mV) (default: -70.6)

        'I_e': 0.,  # Constant external input current (pA) (default: 0.0)
        # Spike adaptation parameters:
        'a': 4.0,  # Subthreshold adaptation (nS) (default: 4.0)
        'b': 80.5,  # Spike-triggered adaptation (pA) (default: 80.5)
        'tau_w': 144.0,  # Adaptation time constant (ms) (default: 144.0)
        'V_th': -50.4,  # Spike initiation threshold (mV) (default: -50.4)
        'V_peak': 0.0,  # Spike detection threshold (mV) (default: 0.0)
        # Exponential spike parameters:
        'Delta_T': 2.0,  # Slope factor (mV) (default: 2.0)
    }

    # The following params come from fitting single cells
    filename = track.ExperimentTracker().get_processed_data_path('fitted_cell_properties.csv')
    cell_props = pd.read_csv(filename, index_col=0)
    default_params = networks.merge_dicts(default_params, {
        # Leak conductance (nS) (fitted: 4.186989, default: 30.0)
        'g_L': cell_props.g_L.median(),
        # Capacitance of the membrane (pF) (fitted: 239.843077, default: 281.0)
        'C_m': cell_props.C_m.median(),
        # Derived
        # R_in = 0.239 (GOhms)
        # tau_m = 53.99 (ms)
    })

    params = {}
    for group_id in ['e', 'i']:
        for k, v in default_params.items():
            params[f'{group_id}_{k}'] = v

    return params


def get_base_params_network():
    params = {}

    # The following params come from fitting single synapses
    filename = track.ExperimentTracker().get_processed_data_path('fitted_connections_dists.csv')
    dists = pd.read_csv(filename).set_index(['param'])

    # TODO split ex & in ?
    params = networks.merge_dicts(params, {
        'tau_syn_ex': dists.loc['tau_syn', 'location'],  # 1.01406 (ms) previously: 1.34
        'tau_syn_in': dists.loc['tau_syn', 'location'],  # 1.01406 (ms) previously: 3.29
    })

    # note that we are later on using this for scipy.stats.uniform which draws in [low, high)
    # (dropping the right-most value from the range).
    # we aftewards round to 1 decimal because that's the resolution that we usually work at with nest
    delays_range = (.45, 2.1)
    params = networks.merge_dicts(params, {
        'delays_low': delays_range[0],
        'delays_high': delays_range[1],
    })

    # The following params come from Mike's estimations

    rev_potentials = get_reversal_potentials()
    params = networks.merge_dicts(params, {
        'E_ex': rev_potentials['E_ex'],  # Excitatory reversal potential (mV) (default: 0.0)
        'E_in': rev_potentials['E_in'],  # Inhibitory reversal potential (mV) (default: -85.0)
    })

    # The following params come from a mix of Mike's estimations and fitting the resting potential
    #
    # We want to use the same distribution for J_in as for J_ex because we
    # can't trust IPSP meassurements (because they were made at reversal potential)
    #
    # From voltage clamp recordings we can estimate that i-currents are
    # 2-3 times that of e-currents induced by synapses.
    v_clamp_in = -50.
    v_clamp_ex = -70.
    ratio = 2.5

    balanced_ratio = (params['E_ex'] - v_clamp_ex) / (params['E_in'] - v_clamp_in)
    g = ratio * balanced_ratio

    # note that we will be scaling the distribution by -g in the init code
    params = networks.merge_dicts(params, {
        # Synaptic parameters
        'g': g,  # -8 previously 7.2
    })

    # The following params come from fitting single synapses
    params = networks.merge_dicts(params, {
        'e_weights_loc': dists.loc['weight', 'location'],
        'e_weights_scale': dists.loc['weight', 'scale'],
        'e_weights_shape': dists.loc['weight', 'shape'],
        'e_weights_vmax': dists.loc['weight', 'vmax'],
    })

    params['e_weights_range_low'] = 0.
    params['e_weights_range_high'] = 1.

    # The network size!
    params = networks.merge_dicts(params, {
        'N': 100000,
    })

    # The probability of connection come from Mike's pair tests
    probs_filename = track.ExperimentTracker().get_external_data_path('connection_probs.csv')
    conn_probs = pd.read_csv(probs_filename, index_col=0)
    params = networks.merge_dicts(params, {
        n + '_probability': v
        for n, v in conn_probs['P(mono)'].items()})

    params = networks.merge_dicts(params, get_connectivity_profiles())

    # The following params are hand-picked just for random initialization of the network
    params = networks.merge_dicts(params, {
        'V_0_mean': -60.0,  # (mV)
        'V_0_std': 4.0,  # (mV)
        'w_0_mean': 100.0,  # (pA)
        'w_0_std': 4.0,  # (pA)
    })

    return params
