#!/usr/bin/env python

""" 
    Slope layer class

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

class SlopeLayer(BaseLayer):

    def __init__(
        self, fpath: str, units, max_val: float=None) -> None:
        """ Init slope layer
        
        Args:
            fpath: absolute path to .tif raster
            units: units of the raster data ('deg' or 'rad' supported)
            max_val: max slope (in layer units)
        """
        super().__init__(fpath)

        self.fpath = fpath
        self.units = units
        self.max_val = max_val
        
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
            if self.max_val is not None:
                self.max_val = np.deg2rad(self.max_val)
        elif self.units == 'rad':
            pass
        else:
            raise ValueError(f"Unsupported slope layer units: {self.units}")
    
    def get_raster(self) -> np.array:
        """Return processed raster"""
        return self._raster
    
    def update_nogo(self) -> None:
        """Nogo wherever the slope is too steep"""
        self._nogo = np.zeros(self.get_raster().shape)
        if self.max_val is not None:
            self._nogo[self.get_raster() > self.max_val] = 1
    
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
        
        ax.set_title(f"Slope ({self.units})", y=1.05)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    
