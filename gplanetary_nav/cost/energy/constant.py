#!/usr/bin/env python

""" 
    Constant power consumption model

    Authors: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import logging

from gplanetary_nav.site.loader import SiteLoader
from gplanetary_nav.cost.base_cost import BaseCost

log = logging.getLogger(__name__)

class ConstantPower(BaseCost):

    # def __init__(self, site: SiteLoader, rover_cfg: dict):
    #     """Init constant velocity model

    #     Args:
    #         site: site instance
    #         rover_cfg: rover configuration dictionary
    #     """
    #     super().__init__(site, rover_cfg)

    def eval(self, node, pitch, roll, length, duration=None) -> float:
        """Energy expenditure to cross an edge segment
        
        Args:
            (See BaseCost.eval for full documentation)

        Return:
            energy consumed, in Whr
        """

        if duration is None:
            raise ValueError("Traverse duration not provided")
        
        # Convert time to hours
        return self.rover_cfg['motion']['power']*duration/3600