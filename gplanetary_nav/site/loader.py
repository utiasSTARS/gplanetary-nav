#!/usr/bin/env python

""" 
    Load overhead georeferenced terrain maps of a site

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto

    A special thanks to Kyohei Otsu (NASA Jet Propulsion Laboratory, California 
    Institute of Technology), whose work inspired the creation of this module.
"""

from dataclasses import dataclass
import functools
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Set, Union
from typing import Callable

from scipy import ndimage
import pandas as pd
import numpy as np
import rasterio
import rasterio.mask
import rasterio.plot
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import cm

from gplanetary_nav.planning.types import Node
from gplanetary_nav.utils import remove_islands, load_yaml_with_includes
from gplanetary_nav.site.layers import *


log = logging.getLogger(__name__)

@dataclass
class StdLayerMeta:
    """Standard layer name and type"""
    name: str
    type: type


def load(dirpath: str) -> object:
    """Load site by its absolute directory path"""
    if not os.path.exists(dirpath):
        log.error(f"Configuration not found: {dirpath}")
        return None

    return SiteLoader(dirpath)

class SiteLoader(object):

    # Standard layer names and types from layer code
    std_layer_meta = {
        'dem': StdLayerMeta('dem', DEMLayer),
        'slope': StdLayerMeta('slope', SlopeLayer),
        'aspect': StdLayerMeta('aspect', AspectLayer),
        'cfa': StdLayerMeta('cfa', CFALayer),
        'terrain': StdLayerMeta('terrain', TerrainLayer),
        'nogo': StdLayerMeta('nogo', NogoLayer)
    }

    def __init__(self, dirpath: str):
        """Load Site and all related maps

        Args:
            dirpath: absolute path to dataset directory containing geotiff 
                files and settings.yaml
        """

        if not os.path.exists(dirpath):
            log.error(f"Configuration not found: {dirpath}")
            raise

        self.dirpath = dirpath
        self.base_name = None   # Name of the reference (base) standard layer

        self.layers = dict()        # all layers except 'other' layers
        self.other_layers = dict()  # all 'other' layers

        self.nogo_kernel = None
        self.tempfiles = []

        # Load site settings
        try:
            self.load_data()
        except Exception as e:
            log.exception(f"Problem while loading dataset: {e}")
            raise

        log.info(f"Dataset {self.name} was loaded")
        self.create_utm_meshgrids()        
        
    def __del__(self):
        if self.tempfiles:
            log.debug("Removing temporary files")
            for fn in self.tempfiles:
                try:
                    os.remove(fn)
                except OSError:
                    pass

    def load_data(self):
        """Load dataset stored in a directory."""
        log.info(f"Loading dataset from: {self.dirpath}")

        # Load site settings
        fpath = Path(self.dirpath, 'settings.yaml')
        settings = load_yaml_with_includes(fpath)
        
        # Site basic attributes
        self.name = os.path.basename(self.dirpath)
        self.center_longitude = settings['center_longitude']
        self.center_latitude = settings['center_latitude']
        self.reference_body = settings['reference_body'].lower()

        # no-go processing parameters
        if settings['nogo_processing']['inflation_kernel_dim'] is not None:
            self.nogo_kernel = np.ones((
                settings['nogo_processing']['inflation_kernel_dim'],
                settings['nogo_processing']['inflation_kernel_dim']))
        
        self.free_island_is_nogo = settings['nogo_processing']['free_island_is_nogo']
        self.islands_corners_connect = settings['nogo_processing']['islands_corners_connect']
        self.border_is_nogo = settings['nogo_processing']['border_is_nogo']
        
        # Standard layers
        self.layer_code_from_name = dict()
        for l_code, l_attrs in settings['layers'].items():
            if l_code not in ['nogo', 'other']:

                log.info(f"Loading {l_code}")

                if l_code not in self.std_layer_meta.keys():
                    raise ValueError(f"Invalid layer code: {l_code}")
                
                l_name = l_attrs['fpath'].split('.')[0]
                self.std_layer_meta[l_code].name = l_name
                self.layer_code_from_name[l_name] = l_code

                l_attrs['fpath'] = os.path.join(self.dirpath, l_attrs['fpath'])
                self.layers[l_name] = self.std_layer_meta[l_code].type(**l_attrs)

                if self.base_name is None:
                    self.base_name = l_name
            
            elif l_code == 'nogo':
                for fpath in l_attrs:
                    log.info(f"Loading {fpath}")
                    l_name = fpath.split('.')[0]
                    if l_name == 'nogo':
                        raise ValueError(f"Cannot name no-go file 'nogo.tif'")

                    fpath = os.path.join(self.dirpath, fpath)
                    self.layers[l_name] = self.std_layer_meta[l_code].type(fpath)
        
        # Other layers
        try:
            for fpath in settings['layers']['other']:
                l_name = fpath.split('.')[0]
                fpath = os.path.join(self.dirpath, fpath)
                self.other_layers[l_name] = BaseLayer(fpath)
        except KeyError:
            log.info(f"No 'other' layers provided")
        
        all_layers = list(self.layers.keys()) + list(self.other_layers.keys())
        log.info(f"Loading site {self.name} with layers {all_layers} "
                 f"at lat-lon ({self.center_latitude:.3f}, {self.center_longitude:.3f}) "
                 f"deg. on reference body: {self.reference_body}")
        
        # Combine bad terrain classes and all no-go maps into one
        self.combine_nogo()
        
    def combine_nogo(self):
        """(Re-)combine all nogo maps into the nogo map attribute"""
        
        if 'nogo' in self.layers:
            self.remove_layer('nogo')
        
        self.combined_nogo_raster = np.zeros(self.base.shape, dtype=np.uint8)

        for layer in self.layers.values():
            self.combined_nogo_raster[layer.get_nogo() == 1] = 1
        
        # Inflate no-go regions
        if self.nogo_kernel is not None:
            self.combined_nogo_raster = ndimage.morphology.binary_dilation(
                self.combined_nogo_raster.astype(np.bool), self.nogo_kernel).astype(np.uint8)

        if self.free_island_is_nogo:
            self.combined_nogo_raster = remove_islands(
                self.combined_nogo_raster,
                self.islands_corners_connect)
        
        # Do not go on map borders
        if self.border_is_nogo:
            self.combined_nogo_raster[[0,-1],:] = 1
            self.combined_nogo_raster[:,[0,-1]] = 1
        
        # Add combined map to new no-go layer
        gtif_meta = self.base.meta
        gtif_meta['nodata'] = None
        gtif_meta['dtype'] = self.combined_nogo_raster.dtype

        self.layers['nogo'] = NogoLayer.from_raster(
            self.combined_nogo_raster, gtif_meta
        )

    def set_nogo_inflation(self, kernel_dims: int=None) -> None:
        """Set the no-go map inflation kernel
        
        Args:
            kernel_dims: dimension (n) of an nxn kernel of ones. None for no
                inflation
        """

        if kernel_dims is not None:
            self.nogo_kernel = np.ones((kernel_dims, kernel_dims))
        else:
            self.nogo_kernel = None

        self.combine_nogo()
    
    def new_layer_from_existing(
        self, in_name: str, out_name: str, mapping_f: Callable[[np.array], np.array],
        new_params: Dict[str,Any]=None) -> None:
        """ Map the values of an integer-valued layer to create a new layer
        
        Args:
            in_name: the name of the (existing) layer
            out_name: the name of the target/output layer. Cannot be a layer
                that's currently loaded. Hint: self.unload_layer() could help
            new_params: the parameters of the new layer (all parameters
                fed to the layer class' init function, except the file name)
            mapping_f: mapping function from the input layer raster to the 
                output raster.
        """

        if out_name in self.layers:
            raise ValueError(f"Layer '{out_name}' already exists")
        
        new_params = dict() if new_params is None else new_params
        out_layer = self.layers[in_name].mapped_copy(mapping_f, **new_params)

        self.layers[out_name] = out_layer

        # Update combined nogo map
        self.combine_nogo()
    
    def remove_layer(self, layer):
        """Clear a layer from memory"""
        del self.layers[layer]
        self.combine_nogo()
    
    @property
    def base(self):
        return self.layers[self.base_name].gtif
    
    @property
    def shape(self) -> tuple:
        return self.base.shape

    @property
    def resolution_mtr(self) -> float:
        """Resolution in meters."""
        return self.base.res[0]

    @property
    def extent(self) -> list:
        """Return Matplotlib plotting extent [left, right, bottom, top]"""
        return rasterio.plot.plotting_extent(self.base)
    
    # QUICK ACCESS TO STANDARD LAYERS
    @property
    def dem(self) -> np.array:
        try:
            return self.layers[self.std_layer_meta['dem'].name].get_raster()
        except KeyError:
            raise
    
    @property
    def slope(self) -> np.array:
        try:
            return self.layers[self.std_layer_meta['slope'].name].get_raster()
        except KeyError:
            raise
    
    @property
    def aspect(self) -> np.array:
        try:
            return self.layers[self.std_layer_meta['aspect'].name].get_raster()
        except KeyError:
            raise
    
    @property
    def cfa(self) -> np.array:
        try:
            return self.layers[self.std_layer_meta['cfa'].name].get_raster()
        except KeyError:
            raise
    
    @property
    def terrain(self) -> np.array:
        try:
            return self.layers[self.std_layer_meta['terrain'].name].get_raster()
        except KeyError:
            raise
    
    @property
    def nogo(self) -> np.array:
        return self.layers['nogo'].get_raster()

    @property
    @functools.lru_cache(maxsize=1)
    def valid_coords(self) -> Set[Node]:
        """Set of coordinates in free (non-nogo) locations"""
        return set(map(tuple, np.argwhere(self.nogo==0).tolist()))

    @property
    def X(self) -> np.array:
        """Return X UTM coordinates grid."""
        return self.X_UTM
    
    @property
    def Y(self) -> np.array:
        """Return Y UTM coordinates grid."""
        return self.Y_UTM
    
    @property
    def roi_UTM_xlim(self) -> tuple:
        """Horizontal (x) UTM limits of a ROI (meters) as a 2-tuple
        
        The ROI is the smallest rectangle that fits all free (i.e. non-nogo)
        pixels. If no 'nogo' layer exists, the full map limits are returned.
        """
        
        roi_layer = self.nogo
        roi_coords = np.argwhere(roi_layer==0)

        xmin = np.min(roi_coords[:,1])
        xmax = np.max(roi_coords[:,1])

        xmin_utm = self.X[0,xmin]
        xmax_utm = self.X[0,xmax]

        return (xmin_utm, xmax_utm)
    
    @property
    def roi_UTM_ylim(self) -> tuple:
        """Horizontal (y) UTM limits of a ROI (meters) as a 2-tuple
        
        The ROI is the smallest rectangle that fits all free (i.e. non-nogo)
        pixels. If no 'nogo' layer exists, the full map limits are returned
        """

        roi_layer = self.nogo
        roi_coords = np.argwhere(roi_layer==0)

        ymin = np.min(roi_coords[:,0])
        ymax = np.max(roi_coords[:,0])

        ymin_utm = self.Y[ymin,0]
        ymax_utm = self.Y[ymax,0]

        return (ymax_utm, ymin_utm)
    
    def plot_roi_rectangle(self, ax: Axes, **kwargs) -> Axes:

        xlims = self.roi_UTM_xlim
        ylims = self.roi_UTM_ylim
        w_rect = xlims[1] - xlims[0]
        h_rect = ylims[1] - ylims[0]
        ax.add_patch(plt.Rectangle(
            (xlims[0], ylims[0]), w_rect, h_rect,
            edgecolor='w', fill=False, zorder=10, **kwargs))
        
        return ax

    @property
    def layers_with_units(self) -> dict:
        """Site layer names and their units"""
        layer_units = dict()
        for l_name, layer in self.layers.items():
            layer_units[l_name] = layer.units
        return layer_units
    
    def plot_all(self, layer_kwargs: Dict[str,dict]=None,
        incl_other: bool=True) -> None:
        """Plot all layers and their no-go counterparts
        
        Args:
            layer_kwargs: a dictionary indexed by layer name and containing
                ax.imshow() kwargs for that specific layer.
            incl_other: whether to plot 'other' layers too
        """

        default_kwargs = {
            'dem': {
                'cmap': cm.coolwarm,
            },
            'slope': {
                'cmap': cm.gray,
            },
            'aspect': {
                'cmap': cm.jet,
            },
            'cfa': {
                'cmap': cm.gray,
            },
            'terrain': {},
            'nogo': {},
        }

        # Add in whatever args specified by the user
        layer_kwargs = dict() if layer_kwargs is None else layer_kwargs
        for l_code, kwargs in layer_kwargs.items():
            for k, v in kwargs.items():
                default_kwargs[l_code][k] = v

        # Plot!
        n = len(self.layers) if not incl_other else len(self.layers) + len(self.other_layers)
        fig, axes = plt.subplots(n,2, figsize=(11,n*4), dpi=150, squeeze=False)
        for i, (l_name, layer) in enumerate(self.layers.items()):
            try:
                layer.plot(axes[i,0], **default_kwargs[self.layer_code_from_name[l_name]])
            except KeyError:
                layer.plot(axes[i,0])
            NogoLayer.plot_static(layer.get_nogo(), axes[i,1], self.extent)

            axes[i,0].annotate(l_name, xy=(0, 0.5), xytext=(-axes[i,0].yaxis.labelpad - 5, 0),
                xycoords=axes[i,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', weight='bold')

        for ax, col in zip(axes[0,:], ['Layer raster', 'Layer no-go']):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, 40),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline', weight='bold')
        
        if incl_other:
            for i, (l_name, layer) in enumerate(self.other_layers.items(), start=len(self.layers)):
                layer.plot(axes[i,0])
                NogoLayer.plot_static(layer.get_nogo(), axes[i,1], self.extent)

                axes[i,0].annotate(l_name, xy=(0, 0.5), xytext=(-axes[i,0].yaxis.labelpad - 5, 0),
                    xycoords=axes[i,0].yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', weight='bold')

        plt.tight_layout()
        fig.subplots_adjust(left=0.1, top=0.95)
        
    def create_utm_meshgrids(self):
        """Create X-Y grids of UTM coordinates."""
        hmin, hmax, vmin, vmax = self.extent
        cell_size = ((vmax - vmin)/self.base.height,
                     (hmax - hmin)/self.base.width)

        Y = np.linspace(vmin+cell_size[0]/2, vmax-cell_size[0]/2,
                        self.base.height, endpoint=True)
        Y = np.flip(Y)
        X = np.linspace(hmin+cell_size[1]/2, hmax-cell_size[1]/2,
                        self.base.width, endpoint=True)

        self.Y_UTM, self.X_UTM = np.meshgrid(Y, X, indexing='ij')
    
    def UTM_from_grid(self, coord: tuple) -> tuple:
        """Return the UTM coordinates at a given grid coordinate
        
        Args:
            coord: map grid coordinates tuple as (row,col)
                row and col can be floating-point numbers
        
        Returns:
            tuple: X,Y UTM coordinates
        """

        row_grid_xp = np.arange(self.base.shape[0])
        Y_UTM_fp = self.Y_UTM[:,0]
        y_utm = np.interp(coord[0], row_grid_xp, Y_UTM_fp)

        col_grid_xp = np.arange(self.base.shape[1])
        X_UTM_fp = self.X_UTM[0,:]
        x_utm = np.interp(coord[1], col_grid_xp, X_UTM_fp)

        return (x_utm,y_utm)
    
    def grid_from_UTM(self, easting: float, northing: float) -> tuple:
        """Nearest grid coordinates at a given UTM coordinate
        
        Args:
            easting (float): the UTM easting
            northing (float): the UTM northing
            
        Returns:
            tuple: map grid coordinates tuple as (row,col)
        """

        y_coord = np.nanargmin(np.abs(self.Y_UTM[:,0]-northing))
        x_coord = np.nanargmin(np.abs(self.X_UTM[0,:]-easting))

        return (y_coord, x_coord)
    
    def value_from_UTM(
        self, easting: float, northing: float, layer: str=None):
        """Determine raster values at given UTM coordinates

        Args:
            easting: Easting (meters) in map coordinate frame
            northing: Northing (meters) in map coordinate frame
            layer (str or None): layer to evaluate. If None, a dict of 
                all layer values {layer : value} is returned.
        
        Return:
            float or dict: layer value
        """

        return self.value_from_grid(
            self.grid_from_UTM(easting, northing), layer)
    
    def value_from_grid(self, coord: Node, layer: str=None) -> Union[float,dict]:
        """Determine raster values at given grid coordinates

        Args:
            coord: grid coordinates
            layer (str or None): layer to evaluate. If None, a dict of 
                all layer values {layer : value} is returned.
        
        Return:
            float or dict: layer value
        """

        y_coord, x_coord = coord

        # Retrieve raster values
        if layer is None:
            data_dict = {}
            for l_name, layer in self.layers.items():
                data_dict[l_name] = layer.get_raster()[y_coord, x_coord]
            return data_dict
        else:
            return self.layers[layer].get_raster()[y_coord, x_coord]

    def in_bounds(self, cell: Tuple[int,int]) -> bool:
        """True if cell is a valid image pixel coordinate"""
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def in_nogo(self, cell: tuple) -> bool:
        """True if the cell (image pixel coordinates) is in a no-go region."""
        return self.nogo[cell] == 1

    def value_from_grid_coordinates(self, coordinates: List[Node]) -> pd.DataFrame:
        """Return terrain layer values from a sequence of grid coordinates

        Args:
            coordinates: list of n grid coordinates
        
        Returns:
            pd.Dataframe: (n,d) dataframe, where d is the number of terrain
                layers on this site
        """

        # Layer names & units (serve as dataframe column headers)
        site_headers = []
        for layer, unit in self.layers_with_units.items():
            if unit is None:
                site_headers.append(layer.upper())
            else:
                site_headers.append(layer.upper()+'_'+unit)

        data = np.empty((0,len(site_headers)))

        for coord in coordinates:
            new = np.array(list(self.value_from_grid(coord).values()))
            data = np.vstack((data,new))
        
        return pd.DataFrame(data, columns=site_headers)


    def value_from_UTM_coordinates(self, coordinates: np.array) -> pd.DataFrame:
        """Return terrain layer values from a sequence of UTM coordinates

        Args:
            coordinates: (n,2) array of [easting, northing] UTM coordinates
        
        Returns:
            pd.Dataframe: (n,d) dataframe, where d is the number of terrain
                layers on this site
        """

        # Layer names & units (serve as dataframe column headers)
        site_headers = []
        for layer, unit in self.layers_with_units.items():
            if unit is None:
                site_headers.append(layer.upper())
            else:
                site_headers.append(layer.upper()+'_'+unit)

        data = np.empty((0,len(site_headers)))

        for (easting, northing) in coordinates:
            new = np.array(list(self.value_from_UTM(easting, northing).values()))
            data = np.vstack((data,new))
        
        return pd.DataFrame(data, columns=site_headers)
    
    def raster_dict(self, layers: List[str]=None) -> dict:
        """ Dictionary of data layers (as np arrays) except 'other' layers

        Args:
            layers: list of layer names to include in the dict.
                If None, all layers will be included. Defaults is None

        Returns:
            dict: dictionary of {layer:data_array}
        """

        if layers is None:
            layers = list(self.layers.keys())

        data_dict = dict()
        for l_name in layers:
            try:
                data_dict[l_name] = self.layers[l_name].get_raster()
            except KeyError:
                log.warn(f"Layer {l_name} not found, skipping")
        
        return data_dict
