#!/usr/bin/env python

""" 
    Constant velocity rover model

    Authors: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import logging

from gplanetary_nav.site.loader import SiteLoader
from gplanetary_nav.cost.base_cost import BaseCost

log = logging.getLogger(__name__)


class ConstantVelocity(BaseCost):

    # def __init__(self, site: SiteLoader, rover_cfg: dict):
    #     """Init constant velocity model

    #     Args:
    #         site: site instance
    #         rover_cfg: rover configuration dictionary
    #     """
    #     super().__init__(site, rover_cfg)

    def eval(self, node, pitch, roll, length, duration=None) -> float:
        """Time duration to cross an edge segment
        
        Args:
            (See BaseCost.eval for full documentation)

        Return:
            traverse duration, in seconds
        """

        return length/self.rover_cfg['motion']['velocity']