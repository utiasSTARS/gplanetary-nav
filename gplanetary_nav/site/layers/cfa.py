#!/usr/bin/env python

""" 
    CFA (cummulative (rock) fractional area) layer class

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

class CFALayer(BaseLayer):

    def __init__(
        self, fpath: str, cfa_norm_factor: float=1.0, max_cfa: float=1.0) -> None:
        """ Init cfa layer
        
        Args:
            fpath: absolute path to .tif raster
            cfa_norm_factor: normalization factor to convert the raster to
                values between 0 and 1. Default: 1.0
            max_cfa: the maximum drivable cfa value, between 0 and 1. Default 1
        """
        super().__init__(fpath)

        self.fpath = fpath
        self.cfa_norm_factor = cfa_norm_factor
        self.max_cfa = max_cfa
        
        self.preprocess()
        self.update_nogo()
    
    def preprocess(self) -> None:
        """Normalize raster"""
        self._raster = super().get_raster()
        self._raster = self._raster.astype(np.float)/self.cfa_norm_factor
    
    def get_raster(self) -> np.array:
        """Return processed raster"""
        return self._raster
    
    def update_nogo(self) -> None:
        """Nogo wherever the CFA is too high values"""
        self._nogo = np.zeros(self.get_raster().shape)
        self._nogo[self.get_raster() > self.max_cfa] = 1
    
    def get_nogo(self) -> np.array:
        return self._nogo
    
    def plot(self, ax: Axes, **kwargs) -> Axes:
        """Layer plotting
        
        Args:
            ax: matplotlib ax
            **kwargs: any keyword argument compatible with ax.imshow()
        
        Return:
            matplotlib ax
        """

        im = ax.imshow(
            self.get_raster(),
            extent=rasterio.plot.plotting_extent(self.gtif),
            **kwargs)
        ax.axis('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        
        ax.set_title(f"CFA", y=1.05)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    
