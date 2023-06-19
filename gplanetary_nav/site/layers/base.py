#!/usr/bin/env python

""" 
    Base class of all layer objects

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from __future__ import annotations

import os
import numpy as np
import rasterio
import rasterio.plot

from matplotlib.axes import Axes 
from matplotlib import cm


class BaseLayer:
    def __init__(self, fpath: str) -> None:
        """ Init layer
        
        Args:
            fpath: absolute path to .tif raster
        """

        self.units = None
        self.load_raster(fpath)

    def load_raster(self, fpath: str) -> None:
        """Load .tif raster from absolute filepath"""
        self.fpath = fpath
        if os.path.isfile(fpath):
            self.gtif = rasterio.open(fpath)
        else:
            raise IOError(f"Could not find {fpath}")

    def get_raster(self) -> np.array:
        """Return geotiff raster data as a np.array"""
        if self.gtif.meta['count'] == 3:
            # 3-channel, like a RGB mosaic
            r = self.gtif.read(1)
            b = self.gtif.read(2)
            g = self.gtif.read(3)
            return np.dstack((r,b,g))
        else:
            # only 1 channel
            return self.gtif.read(1)
    
    def get_nogo(self) -> np.array:
        """nogo binary map of current layer. By default, no obstacles"""
        return np.zeros(self.gtif.shape)

    def update_nogo(self) -> None:
        pass

    def plot(self, ax: Axes, **kwargs) -> Axes:
        """Basic raster plot
        
        Args:
            ax: matplotlib ax
            **kwargs: any keyword argument compatible with ax.imshow()
        
        Return:
            matplotlib ax
        """

        if self.gtif.meta['count'] == 1:
            kwargs['cmap'] = cm.gray
        im = ax.imshow(
            self.get_raster(),
            extent=rasterio.plot.plotting_extent(self.gtif),
            **kwargs)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    
    def save(self, fpath: str) -> None:
        """Save the current layer to the provided absolute .tif file path"""
        with rasterio.open(fpath, 'w', **self.gtif.meta) as dst:
            dst.write(self.get_raster(), self.gtif.meta['count'])

