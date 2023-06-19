#!/usr/bin/env python

""" 
    No-Go layer class

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from __future__ import annotations
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio

from matplotlib.axes import Axes 
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gplanetary_nav.site.layers.base import BaseLayer

class NogoLayer(BaseLayer):
    """Numerical binary layer where 0 = free, 1 = no-go (obstacle)"""

    # Recommended (though, not mandatory) geotiff dtype for no-go layers
    recommended_gtiff_dtype = 'uint8'

    def __init__(self, fpath: str) -> None:
        """ Init nogo layer
        
        Args:
            fpath: absolute path to .tif raster
        """
        super().__init__(fpath)

        self.fpath = fpath
    
    @classmethod
    def from_raster(cls, raster: np.array, gtif_meta: dict) -> NogoLayer:
        """Create layer instance from a raster & corresponding geotiff metadata

        Args:
            raster: no-go data with proper shape & dtype
            gtif_meta: metadata of the no-go map
        
        Return:
            no-go layer instance
        """

        with NamedTemporaryFile(suffix='.tif') as f:
            with rasterio.open(f.name, 'w', **gtif_meta) as dst:
                dst.write(raster, gtif_meta['count'])
            
            return cls(f.name)
    
    def get_nogo(self) -> np.array:
        return self.get_raster()
    
    def plot(self, ax: Axes, **kwargs) -> Axes:
        return NogoLayer.plot_static(
            self.get_raster(),
            ax,
            rasterio.plot.plotting_extent(self.gtif),
            **kwargs)

    @staticmethod
    def plot_static(raster: np.array, ax: Axes, extent: tuple, **kwargs) -> Axes:
        """Layer plotting
        
        Args:
            raster: the no-go numerical binary raster (0 = free, 1 = obstacle)
            ax: matplotlib ax
            extent: matplotlib plotting extent (left, right, bottom, top)
            **kwargs: any keyword argument compatible with ax.imshow()
        
        Return:
            matplotlib ax
        """

        kwargs['cmap'] = cm.get_cmap('gray_r', 2)
        kwargs['vmin'] = 0
        kwargs['vmax'] = 1
        im = ax.imshow(raster, extent=extent, **kwargs)
        ax.axis('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[0.25,0.75])
        # cbar = ax.get_figure().colorbar(im, ax=ax, ticks=[0.25,0.75])
        cbar.set_ticklabels(['free', 'no-go'])

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = ax.get_figure().colorbar(im, cax=cax)
        
        ax.set_title(f"Free & No-Go regions", y=1.05)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    
    def mapped_copy(self, mapping_f: function, **new_params) -> NogoLayer:
        """Create a new no-go layer from a mapping function
        
        Args:
            mapping_f: a function taking the current raster as input and
                outputting the new raster. The new raster is expected to have
                the same shape and data type.
            new_params: parameters that NogoLayer.__init__() takes as input
        
        Return:
            A new no-go layer
        """

        new_raster = mapping_f(self.get_raster())
        with NamedTemporaryFile(suffix='.tif') as f:
            with rasterio.open(f.name, 'w', **self.gtif.meta) as dst:
                dst.write(new_raster, self.gtif.meta['count'])
            
            new_params['fpath'] = f.name
            
            return NogoLayer(**new_params)
    
