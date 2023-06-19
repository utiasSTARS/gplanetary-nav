#!/usr/bin/env python

""" 
    Aspect (slope orientation) layer class

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

class AspectLayer(BaseLayer):

    def __init__(
        self, fpath: str, units) -> None:
        """ Init aspect layer
        
        Args:
            fpath: absolute path to .tif raster
            units: units of the raster data ('deg' or 'rad' supported)
        """
        super().__init__(fpath)

        self.fpath = fpath
        self.units = units
        
        self.preprocess()
        self.update_nogo()
    
    def preprocess(self) -> None:
        """Remove negative values, convert to rad if needed"""

        self._raster = super().get_raster()
        
        # Remove invalid values
        self._raster[self._raster < 0] = 0

        if self.units == 'deg':
            # convert to rad
            self._raster = np.deg2rad(self._raster)
            self.units = 'rad'
        elif self.units == 'rad':
            pass
        else:
            raise ValueError(f"Unsupported slope layer units: {self.units}")
    
    def get_raster(self) -> np.array:
        """Return processed raster"""
        return self._raster
    
    def update_nogo(self) -> None:
        """All aspect values are drivable"""
        pass
    
    def get_nogo(self) -> np.array:
        return np.zeros(self.get_raster().shape)
    
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
        
        ax.set_title(f"Aspect ({self.units})", y=1.05)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    
