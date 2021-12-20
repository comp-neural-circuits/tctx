"""
A safe space for different network setups.

Every setup creates one or more groups of cells.
The method "instantiate" will build the nest network and return a dictionary of the form: {group_id: [cell_id]}
The values are these groups of cell ids and the key, a group id.
They also have a constant attribute 'groups' that is a dictionary of {group_id: GroupDescription}
"""

import copy
from abc import abstractmethod


class MutableNetwork(object):
    """keeps track of a series of mutations that we apply to a network before instantiation.
    This stacking allows us for example to define a 'default' network and then change several parameters,
    one at a time, in a nested sweep

    Every "mutation" is defined by a set of fixed values for a subset of the params of the model
    """

    def __init__(self, instantiate, base_params):
        self.instantiate_func = instantiate
        self.mutations = []
        self.mutations.append(base_params)

    def stack(self, params):
        """returns a copy with the stacked mutation. It does NOT modify this object"""
        stacked = copy.deepcopy(self)
        stacked.mutations.append(params)
        return stacked

    def flatten(self):
        result = dict()

        for m in self.mutations:
            result.update(m)

        return result

    def instantiate(self):
        return self.instantiate_func(self.flatten())


class NetworkInstance:
    """
        Class to hold results of instantiating a network.
        cells and connections attributes are pd.DataFrames.
        Optionally, an instance can be implemented using NEST for simulation.
    """
    def __init__(self, params, cells, connections, extra=None):
        self.params, self.cells, self.connections, self.extra = params, cells, connections, extra

    @abstractmethod
    def implement(self):
        pass


def merge_dicts(*dicts):
    """Given a set of dicts, merge them into a new dict as a deep copy."""
    assert set.intersection(*[set(d.keys()) for d in dicts]) == set()

    merged = copy.deepcopy(dicts[0])

    for d in dicts[1:]:
        merged.update(d)

    return merged
