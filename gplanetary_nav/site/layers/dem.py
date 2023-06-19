#!/usr/bin/env python

""" 
    DEM layer class

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import os
import numpy as np
import rasterio

from matplotlib.axes import Axes 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gplanetary_nav.site.layers.base import BaseLayer

class DEMLayer(BaseLayer):

    def __init__(
        self, fpath: str, units: str='m', min_val: float=None,
        max_val: float=None) -> None:
        """ Init elevation (DEM) layer
        
        Args:
            fpath: absolute path to .tif raster
            units: units of the raster data
            min_val: min. elevation (in layer units) of valid pixels
            max_val: max. elevation (in layer units) of valid pixels
        """
        super().__init__(fpath)

        self.fpath = fpath
        self.units = units
        self.min_val = min_val if min_val is not None else -np.inf
        self.max_val = max_val if max_val is not None else np.inf
        
        self.update_nogo()
    
    def get_nogo(self) -> np.array:
        return self._nogo
    
    def update_nogo(self) -> None:
        """Nogo wherever the DEM has unrealistically low/high values"""
        self._nogo = np.zeros(self.get_raster().shape)
        self._nogo[self.get_raster() < -21900] = 1  # heuristic lower threshold

        self._nogo[self.get_raster() < self.min_val] = 1
        self._nogo[self.get_raster() > self.max_val] = 1
    
    def plot(self, ax: Axes, **kwargs) -> Axes:
        """Layer plotting
        
        Args:
            ax: matplotlib ax
            **kwargs: any keyword argument compatible with ax.imshow()
        
        Return:
            matplotlib ax
        """

        raster = self.get_raster()
        raster[self.get_nogo().astype(np.bool)] = np.nan
        im = ax.imshow(
            raster,
            extent=rasterio.plot.plotting_extent(self.gtif),
            **kwargs)
        ax.axis('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        
        ax.set_title(f"Elevation ({self.units})", y=1.05)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    



    
