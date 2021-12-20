"""
    Code to support the creation of network instances.
"""
import logging

from tctx.util import sim_nest
import nest
import numpy as np
import pandas as pd

from tctx.util.profiling import log_time
import tctx.util.networks


class NetworkInstance(tctx.util.networks.NetworkInstance):
    """
        Class to hold results of instantiating a network.
        cells and connections attributes are pd.DataFrames.
        Optionally, an instance can be implemented using NEST for simulation.
    """

    def __init__(self, params, cells, connections, extra=None):
        super().__init__(params, cells, connections, extra)
        report(cells, connections, level=logging.DEBUG)

    def implement(self):
        with log_time('implement nest'):
            self.cells['nest_id'] = self._implement_nest()

        with log_time('initialise nest'):
            self._initialise_nest()

    def _implement_nest(self):
        """translate the self into exectuable nest nodes and connections"""

        nest_ids = self._instantiate_nest_cells()

        self._instantiate_nest_connections(nest_ids)

        return nest_ids

    def _instantiate_nest_cells(self):

        # these are params required by NEST that we may want to differentiate between E and I populations
        group_specific_neuron_keys = {
            't_ref', 'V_reset', 'E_L', 'I_e',
            'a', 'b', 'tau_w', 'V_th', 'V_peak', 'Delta_T',
            'C_m', 'g_L'}

        # these are params required by NEST that are common to all of our neurons
        common_neuron_keys = {
            'E_ex', 'E_in', 'tau_syn_ex', 'tau_syn_in'
        }
        common_neuron_params = {k: self.params[k] for k in common_neuron_keys}

        model = self.params['model']

        groups = {}
        for group_id in ['e', 'i']:
            group_neuron_params = common_neuron_params.copy()

            group_neuron_params.update({
                k: self.params[f'{group_id}_{k}']
                for k in group_specific_neuron_keys
            })

            group_size = np.count_nonzero(self.cells.ei_type == group_id)
            nest_ids = nest.Create(model, group_size, group_neuron_params)
            groups[group_id] = list(nest_ids)  # we want lists because it's what pandas likes to index by

        nest_ids = pd.Series(
            data=np.concatenate([groups['e'], groups['i']]),
            index=np.concatenate([self.cells[self.cells.ei_type == 'e'].index.values,
                                  self.cells[self.cells.ei_type == 'i'].index.values]))

        return nest_ids

    def _instantiate_nest_connections(self, nest_ids):

        sender_gids = nest_ids.reindex(self.connections['source']).values
        receiver_gids = nest_ids.reindex(self.connections['target']).values
        weights = self.connections.weight.values
        delays = self.connections.delay.values

        with log_time(f'create NEST connections ({len(sender_gids):,})', pre=True, level=logging.INFO):
            nest.Connect(
                sender_gids,
                receiver_gids,
                {
                    'rule': 'one_to_one'
                },
                {
                    'model': 'static_synapse',
                    'weight': weights,
                    'delay': delays
                }
            )

    def _initialise_nest(self):
        nodes = list(self.cells.nest_id.values)
        sim_nest.random_init_normal(nodes, 'V_m', self.params['V_0_mean'], spread=self.params['V_0_std'])
        sim_nest.random_init_normal(nodes, 'w', mean=self.params['w_0_mean'], spread=self.params['w_0_std'])


class ByDist:
    """Use a different distribution based on a value"""

    def __init__(self, **dists):
        """
        :param dists: a map from `by` to callable objects
        """
        self.dists = dists

    def sample(self, series):
        """
            Generate a weight for each entry in series.
            Depends only on the property given in series.
        """
        all_values = []

        for group_id, group in series.groupby(series):
            values = self.dists[group_id].sample(len(group))
            all_values.append(pd.Series(data=values, index=group.index))
            logging.debug('created %d %s weights between [%f, %f]',
                          len(values), group_id, np.min(values), np.max(values))

        all_values = pd.concat(all_values)

        return all_values


class ClippedDist:
    """ensure we do not make weights that are ridiculously high.
    This will keep re-sampling untilt `count` elements are generated and all are less or equal to `vmax`.
    """

    def __init__(self, dist, vmin, vmax):
        """
        :param dist: callable that samples a given number
        :param vmax: maximum allowed value
        """
        self.dist = dist
        self.vmax = vmax
        self.vmin = vmin

    # @numba.jit
    def sample(self, count):
        values = np.zeros(count, dtype=np.float)

        # it might be nice to use a product with a sigmoidal in order to avoid the "hard" max
        # TODO apply this also to non-space turtle

        invalid = np.ones(count, dtype=np.bool_)
        left = np.count_nonzero(invalid)

        while left != 0:
            values[invalid] = self.dist(left)

            abs_values = np.abs(values)

            invalid = (abs_values > np.abs(self.vmax)) | (abs_values < np.abs(self.vmin))

            left = np.count_nonzero(invalid)

        return values

    def scale(self, g):
        """given a distribution of excitatory weights, create one that is g-times stronger"""

        def scaled_sample(size):
            sample = self.dist(size)
            return g * sample

        return ClippedDist(dist=scaled_sample, vmax=g * self.vmax, vmin=self.vmin)


class FixedDist:
    """returns always the same value"""

    def __init__(self, constant):
        self.constant = constant

    def sample(self, count):
        return np.ones(count, dtype=np.float) * self.constant

    def scale(self, g):
        """given a distribution of excitatory weights, create one that is g-times stronger"""
        return FixedDist(constant=g * self.constant)


def sample_delays(connections, delay_dist, decimals=1):
    """create delays from the given distribution. Independent from connection properties"""
    if connections.empty:
        return pd.Series(index=connections.index)

    delays = delay_dist(len(connections))
    delays = np.round(delays, decimals=decimals)

    return delays


def report(cells, connections, level):
    with log_time('report', level=level):
        cell_type_counts = cells.groupby('ei_type')['ei_type'].count()
        logging.log(level, 'Cells:\n%s', cell_type_counts)

        connections_type_counts = connections.groupby('con_type')['con_type'].count()

        possible_connections_counts = pd.Series({
            con_type: cell_type_counts[con_type[0]] * cell_type_counts[con_type[-1]] for con_type in connections_type_counts.index
        })

        # it may be a CategoricalIndex in which we can't insert 'all'
        connections_type_counts.index = connections_type_counts.index.astype(str)

        connections_type_counts.loc['all'] = connections_type_counts.sum()
        possible_connections_counts.loc['all'] = np.square(cell_type_counts.sum())

        connections_type_probs = connections_type_counts / possible_connections_counts

        con_stats = pd.DataFrame(
            data={
                'count': connections_type_counts,
                'possible': possible_connections_counts,
                'probs': connections_type_probs,
                'weight mean': connections.groupby('con_type')['weight'].mean(),
                'weight min': connections.groupby('con_type')['weight'].min(),
                'weight max': connections.groupby('con_type')['weight'].max(),
                'delay mean': connections.groupby('con_type')['delay'].mean(),
                'delay min': connections.groupby('con_type')['delay'].min(),
                'delay max': connections.groupby('con_type')['delay'].max(),
            },
            # columns=['count' 'prob.', 'weight_mean', 'weight_std', 'delay_mean', 'delay_std'],
        )

        con_stats.loc['all', 'weight mean'] = connections['weight'].mean()
        con_stats.loc['all', 'weight min'] = connections['weight'].min()
        con_stats.loc['all', 'weight max'] = connections['weight'].max()
        con_stats.loc['all', 'delay mean'] = connections['delay'].mean()
        con_stats.loc['all', 'delay min'] = connections['delay'].min()
        con_stats.loc['all', 'delay max'] = connections['delay'].max()

        logging.log(level, 'Connections:\n%s', con_stats.to_string())
