#!/usr/bin/env python

""" 
    M2020 Terrain Traversability analysis Tools (MTTT) velocity function
    Based on Ono et al. (2018). See Figure 5 for velocity model

    Authors: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import logging
import numpy as np

from gplanetary_nav.site.loader import SiteLoader
from gplanetary_nav.cost.base_cost import BaseCost

log = logging.getLogger(__name__)


class MTTTVelocity(BaseCost):

    def __init__(self, site: SiteLoader, rover_cfg: dict, optimistic: bool=False):
        """Init MTTT velocity model
        
        Args:
            site: site instance
            rover_cfg: rover configuration dictionary
            optimistic: use the fastest velocity estimates instead of the
                slowest ones.
        """
        super().__init__(site, rover_cfg, optimistic=optimistic)

    def eval(self, node, pitch, roll, length, duration=None) -> float:
        """Time duration to cross an edge segment
        
        Args:
            (See BaseCost.eval for full documentation)

        Return:
            traverse duration, in seconds
        """

        # Implementation of Figure 5 in Ono et al. (2018)
        # Note that terrain class indexing starts at 0 in this implementation
        
        if self.site.terrain[node] == 0: # Terrain class 1 in paper
            if (self.site.slope[node] <= np.deg2rad(15) and
                self.site.cfa[node] <= 0.07):
                return length/64.8*3600
            
            elif (  self.site.slope[node] <= np.deg2rad(20) and
                    self.site.cfa[node] <= 0.15):
                return length/52.5*3600
            
            elif (  self.site.slope[node] <= np.deg2rad(25) and
                    self.site.cfa[node] <= 0.15):
                return length/10.9*3600
            else:
                return np.inf
        

        elif self.site.terrain[node] == 1: # Terrain class 2 in paper
            
            if (self.site.slope[node] <= np.deg2rad(20) and
                self.site.cfa[node] <= 0.15):

                if self.optimistic:
                    return length/48.5*3600
                else:
                    return length/24.2*3600
            
            elif (  self.site.slope[node] <= np.deg2rad(25) and
                    self.site.cfa[node] <= 0.15):
                return length/10.9*3600
            else:
                return np.inf

        elif self.site.terrain[node] == 2: # Terrain class 3 in paper
            
            if (self.site.slope[node] <= np.deg2rad(10) and
                self.site.cfa[node] <= 0.15):

                if self.optimistic:
                    return length/40.8*3600
                else:
                    return length/10.9*3600
            else:
                return np.inf
        
        elif self.site.terrain[node] == 3: # Terrain class 4 in paper
            
            if (self.site.slope[node] <= np.deg2rad(10) and
                self.site.cfa[node] <= 0.15):
                return length/10.9*3600
            else:
                return np.inf
        
        elif self.site.terrain[node] in [4,255]: # Terrain class 5 in paper
            return np.inf
        
        else:
            raise ValueError(
                f"Invalid MTTT terrain class: {self.site.terrain[node]} "
                f"(only expects terrain classes 0,1,2,3,4,255)")

