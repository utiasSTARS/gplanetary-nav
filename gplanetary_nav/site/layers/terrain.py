#!/usr/bin/env python

""" 
    Terrain layer class

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from __future__ import annotations
from pathlib import Path
from tempfile import NamedTemporaryFile
import logging
from typing import Iterable, Dict

import yaml
import numpy as np
import rasterio
from matplotlib.axes import Axes 
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from gplanetary_nav.site.layers.base import BaseLayer

log = logging.getLogger(__name__)


class TerrainLayer(BaseLayer):

    def __init__(
        self, fpath: str, labels: Dict[int,str]=None,
        avoid: Iterable[int]=None) -> None:
        """ Init terrain layer
        
        Args:
            fpath: absolute path to .tif raster
            labels: dictionary of terrain label : name. If None, the module
                estimates the number of classes from the raster (not ideal)
            avoid: terrain labels/classes to avoid/ignore when plotting
        """
        super().__init__(fpath)

        self.fpath = fpath
        self.labels = labels
        self.avoid = set() if avoid is None else set(avoid)
        
        self.update_nogo()
    
    def update_nogo(self) -> None:
        """Nogo wherever the terrain needs to be avoided"""
        self._nogo = np.zeros(self.get_raster().shape)
        for bad_label in self.avoid:
            self._nogo[self.get_raster() == bad_label] = 1
    
    def get_nogo(self) -> np.array:
        return self._nogo
    
    @property
    def num_classes(self) -> int:
        """Number of defined terrain classes (everything except 255)"""
        if self.labels is not None: # Preferred. Follows provided labels
            tmp = set(self.labels.keys())
            tmp.remove(255)
            return len(tmp)
        else: # Infer the number of classes, though this is not ideal
            unique_vals = np.unique(self.get_raster())
            unique_vals = np.delete(unique_vals, np.argwhere(unique_vals==255))

            # Assume that the labels are indexed starting from 0, and that the
            # highest label is in the array
            return unique_vals[-1]+1
    
    @property
    def valid_labels(self):
        return np.arange(start=0, stop=self.num_classes)

    def plot(self, ax: Axes, **kwargs) -> Axes:
        """Layer plotting. Mask terrain classes to avoid
        
        Args:
            ax: matplotlib ax
            **kwargs: any keyword argument compatible with ax.imshow()
        
        Return:
            matplotlib ax
        """

        # Mask terrain values to avoid
        terrain_ma = np.ma.masked_where(self.get_nogo(), self.get_raster())
        # if np.ma.is_masked(terrain_ma):
        #     unique_ma = np.unique(terrain_ma)
        #     unique_valid = [l for l, m in zip(unique_ma.data, unique_ma.mask) if not m]
        # else:
        #     unique_valid = np.unique(terrain_ma)

        # Valid categories to show
        # valid_categories = np.arange(start=unique_valid[0], stop=unique_valid[-1]+1)

        kwargs['cmap'] = cm.get_cmap('viridis', len(self.valid_labels))
        kwargs['vmin'] = 0
        kwargs['vmax'] = self.valid_labels[-1]
        im = ax.imshow(
            terrain_ma,
            extent=rasterio.plot.plotting_extent(self.gtif),
            **kwargs)
        ax.axis('equal')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        # Center ticks on corresponding color in colorbar
        tick_locs = (np.arange(len(self.valid_labels)) + 0.5)\
            *(len(self.valid_labels)-1)/len(self.valid_labels)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(self.valid_labels)
        

        ax.set_title(
            f"Terrain classes \n(labels {self.avoid if len(self.avoid) > 0 else '[]'} are masked)", y=1.05)
        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")

        return ax
    
    def mapped_copy(self, mapping_f: function, **new_params) -> TerrainLayer:
        """Create a new terrain layer using a mapping function
        
        Args:
            mapping_f: a function taking the current raster as input and
                outputting the new raster. The new raster is expected to have
                the same shape and data type as the input.
            new_params: parameters fed to TerrainLayer.__init__() for the new
                raster. All but fpath is necessary.
        
        Return:
            A new terrain layer
        """

        new_raster = mapping_f(self.get_raster())
        with NamedTemporaryFile(suffix='.tif') as f:
            with rasterio.open(f.name, 'w', **self.gtif.meta) as dst:
                dst.write(new_raster, self.gtif.meta['count'])
            
            new_params['fpath'] = f.name
            
            return TerrainLayer(**new_params)
    
    def save(self, fpath: Path) -> None:
        """Save the current layer to the provided absolute .tif file path"""
        with rasterio.open(fpath, 'w', **self.gtif.meta) as dst:
            dst.write(self.get_raster(), self.gtif.meta['count'])
        
        if self.labels is not None:
            # Save labels too
            lpath = Path(fpath.parent, fpath.stem + '_labels.yaml')
            with open(lpath, 'w') as f:
                yaml.dump(self.labels, f)


