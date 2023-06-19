#!/usr/bin/env python

""" 
    Numba-optimized solar energy generation calculation

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import numba
import numpy as np

# Numba dict storing power timeseries at every location/node
tuple_type = numba.types.UniTuple(numba.types.int64, 2)
arr_type = numba.types.float32[:,:]

node_power_numba_d = numba.typed.Dict.empty(
    tuple_type,
    arr_type,
)

spec = [
    ('node_power', numba.typeof(node_power_numba_d))
]

@numba.experimental.jitclass(spec)
class NumbaEnergy(object):

    def __init__(self, node_power):
        """Init
        
        Args:
            node_power: dictionary storing a piecewise constant (step function)
                power timeseries for each node/location.
                Every timeseries is assumed to be a (n,2) array, where each row
                represents the (x,y) start of a step, which extends towards +x
                (i.e. 'right-continuous'). Timeseries units are seconds, Watts.
        """
        self.node_power = node_power
    
    def diff1D(self, arr):
        """Numba-optimized equivalent of np.diff on a 1-D array"""
        return arr[1:]-arr[:-1]
    
    def energy(self, node, t_bounds) -> float:
        """Solar energy via piecewise power timeseries integration
        
        Args:
            node: the location (pixel coords) in which the action is taken
            t_bounds: (start time, end time) bounds, each a unix timestamp (s)
        
        Return:
            float: energy produced (Whr)
        """
        power_intervals = self.node_power[node]

        idx_min = np.searchsorted(power_intervals[:,0], t_bounds[0])-1
        idx_max = np.searchsorted(power_intervals[:,0], t_bounds[1])-1

        if idx_min == idx_max:
            # (/3600 to convert duration from s to hr)
            return power_intervals[idx_min,1]*(t_bounds[1]-t_bounds[0])/3600

        energy = 0
        energy += (power_intervals[idx_min+1,0]-t_bounds[0])*power_intervals[idx_min,1]
        energy += (t_bounds[1]-power_intervals[idx_max,0])*power_intervals[idx_max,1]

        if idx_min+1 == idx_max:
            return energy/3600 # convert W-s to W-hr
        else:
            energy += np.dot(
                self.diff1D(power_intervals[idx_min+1:idx_max+1,0]),
                power_intervals[idx_min+1:idx_max,1])

            return energy/3600 # convert W-s to W-hr