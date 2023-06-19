#!/usr/bin/env python

""" 
    Base cost class (with documentation)

    Authors: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from gplanetary_nav.planning.types import Node
from gplanetary_nav.site.loader import SiteLoader

class BaseCost(object):
    """Base class for edge cost calculation"""

    def __init__(self, site: SiteLoader, rover_cfg: dict, **params) -> None:
        """Initialize a BaseCost instance
        
        Args:
            site: the site instance
            rover_cfg: the rover configuration dictionary
            params: additional parameters to save as class attributes
        """

        self.site = site
        self.rover_cfg = rover_cfg
        for k, v in params.items():
            setattr(self, k, v)

    def eval(
        self, node: Node, pitch: float,
        roll: float, length: float, duration: float=None) -> float:
        """Cost of a drive in the given rover & site config
        
        Args:
            node: pixel grid coordinates where drive occurs in site
            pitch: rover pitch in radians (see docs for sign convention)
            roll: rover roll in radians (see docs for sign convention)
            length: length of the edge segment in meters
            duration: traverse time of the edge segment, in seconds (optional)
        """
        pass
