#!/usr/bin/env python

"""
    Random surface maps & dataset generation

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import os
from pathlib import Path
import random
import logging
import yaml
import numpy as np
import rasterio
import gdal

from skimage.draw import ellipse
from hkb_diamondsquare import DiamondSquare as DS

from gplanetary_nav.external.perlin2d import generate_perlin_noise_2d

log = logging.getLogger(__name__)


class SiteGenerator:
    def __init__(self, shape, raw_dem=None, method="DS", **kwargs):
        """Generate a (unitless) raw elevation map.

        Args:
            shape ((int, int)): Terrain height and width tuple, in pixels.
            method (str, optional): Method to generate the height map.
                'DS' = Diamond Squares (default), 'PE' = Perlin Noise.
            raw_dem (ndarray): 2D raw elevation array. If not None, this 
                elevation map will be used instead of generating a new one.
            **kwargs: Generation method-specific parameters. See notes below.
        
        Returns:
            TerrainGenerator: instance for terrain generation.
        
        Notes:
            **kwargs depend on the generation method.
            Method 'DS' expects:
                roughness (float): 0 (smooth) and 1 (rough). Default is 0.5
            Method 'PE' expects:
                per ((int,int)): Num. periods of noise along y-axis and x-axis
                    Default is 5 along each axis
                til ((bool,bool)): Whether to tile along y-axis or x-axis
                    Default is False for each axis
        """

        # Name of layer attributes of interest (anything but the raw_map...)
        self.layers = set()

        # Create an empty class instance if no shape is passed
        if raw_dem is not None:
            self.shape = raw_dem.shape
            self.hwratio = float(self.shape[0])/float(self.shape[1])
            self.Z = raw_dem
            return

        self.shape = shape
        self.hwratio = float(self.shape[0])/float(self.shape[1])
        
        # Generate the raw elevation map
        if method == "DS":
            roughness = kwargs['roughness'] if 'roughness' in kwargs else 0.5

            self.Z = DS.diamond_square(shape=shape, 
                                        min_height=-1.0, 
                                        max_height=1.0,
                                        roughness=roughness)

        elif method == "PE":
            per_y = kwargs['per'][0] if 'per' in kwargs else 5
            per_x = kwargs['per'][1] if 'per' in kwargs else 5
            til_y = kwargs['til'][0] if 'til' in kwargs else False
            til_x = kwargs['til'][1] if 'til' in kwargs else False

            self.Z = generate_perlin_noise_2d(shape, (per_y,per_x), (til_y,til_x))
        
        else:
            raise NotImplementedError(f"Unknown method: {method}. "
                                       "Valid methods: 'DS' or 'PE'")
    
    # @classmethod
    # def from_raw_dem(cls, raw_dem):
    #     """Create a terrain generator from a raw (unitless) elevation map.
        
    #     Args:
    #         raw_dem (ndarray): 2D raw elevation array. If not None, this
    #             elevation map will be used instead of generating a new one.
        
    #     Returns:
    #         TerrainGenerator: instance with provided raw_map
    #     """

    #     return cls(None, raw_dem=raw_dem)
    
    @property
    def raw_dem(self):
        """Return raw (unitless) dem."""
        return self.Z
    

    def create_geometric_maps(self,
            res: float, scale: float, tmp_dir: Path=Path("/tmp")):
        """Create a metric elevation map and slope-aspect maps.

            Args:
                res: the desired resolution (m/px) of the raw map.
                scale: by how much to scale the raw map unitless 
                    elevation values to obtain an elevation map, in metres. 
                tmp_dir (str): Complete path where to save the temporary
                    maps (elevation, slope, aspect) created. If none, the 
                    current directory is used.
            
            Returns:
                (ndarray, ndarray, ndarray): the elevation, slope and aspect 
                    maps, respectively. The elevation map is in meters, the 
                    slope and aspect maps are in degrees.
        """

        # Create map meshgrid
        x = np.linspace(0.,1.0,self.shape[1])
        y = np.linspace(0.,self.hwratio,self.shape[0])
        xg, yg = np.meshgrid(x, y)

        # Meshgrid and elevation map in meters
        xg_dem = xg*xg.shape[1]*res
        self.dem = self.Z*scale
        
        # Path to temporary files
        dem_fpath = Path(tmp_dir, 'dem_tmp.tif')
        slope_fpath = Path(tmp_dir, 'slope_tmp.tif')
        aspect_fpath = Path(tmp_dir, 'aspect_tmp.tif')

        # Save the DEM as a geotiff file
        self.map_transform = rasterio.transform.from_bounds(
            -xg_dem[0,-1]/2.0,
            -self.hwratio*xg_dem[0,-1]/2.0,
            xg_dem[0,-1]/2.0,
            self.hwratio*xg_dem[0,-1]/2.0,
            self.shape[1],
            self.shape[0]
        )

        with rasterio.open(
            dem_fpath,
            'w',
            driver='GTiff',
            height=self.shape[0],
            width=self.shape[1],
            count=1,
            dtype=self.dem.dtype,
            crs=None,
            transform=self.map_transform
        ) as dst:
            dst.write(self.dem, 1)
        
        # Generate slope.tif from dem.tif
        gdal.DEMProcessing(slope_fpath.as_posix(), dem_fpath.as_posix(), 'slope')
        with rasterio.open(slope_fpath) as dataset:
            self.slope=dataset.read(1)
            self.slope[self.slope < 0] = 0  # remove invalid data

        # Generate aspect.tif from dem.tif
        gdal.DEMProcessing(aspect_fpath.as_posix(), dem_fpath.as_posix(), 'aspect')
        with rasterio.open(aspect_fpath) as dataset:
            self.aspect=dataset.read(1)
            self.aspect[self.aspect < 0] = 0  # remove invalid data
        
        # Delete temporary files created
        os.remove(dem_fpath)
        os.remove(slope_fpath)
        os.remove(aspect_fpath)

        # Set map extent for plotting
        self._plotting_extent = [0,xg_dem[0,-1],self.hwratio*xg_dem[0,-1],0]

        log.info(f"Map shape: {self.dem.shape[0]}x{self.dem.shape[1]} px")
        log.info(f"Map resolution: {res} m/px")
        log.info(f"Map dimensions: {xg_dem[0,-1]}x{self.hwratio*xg_dem[0,-1]} m")
        log.info(f"Maximum slope: {self.slope.max():.2f} deg")

        # Update layers list
        self.layers.add('dem')
        self.layers.add('slope')
        self.layers.add('aspect')

        return self.dem, self.slope, self.aspect
    
    def create_terrain_classes(self, num_clusters,
                               num_classes, max_cluster_width):
        """Create a random terrain class map using elliptical terrain clusters.

        Args:
            num_clusters (int): number of terrain class clusters to randomly 
                sample across the map.
            num_classes (int): number of possible different terrain classes.
            max_cluster_width (int): maximum width (px) of a single cluster.
        
        Return:
            ndarray: terrain class map, where every pixel is
            labelled 0,1,...,num_classes-1
        """

        self.num_terrain_classes = num_classes

        self.terrain = np.zeros(self.shape, dtype=np.uint8)

        for i in range(num_clusters):
            # Sample cluster center
            r = random.random()*self.shape[0]
            c = random.random()*self.shape[1]

            # Sample cluster axes and rotation
            r_radius = random.random()*max_cluster_width
            c_radius = random.random()*max_cluster_width
            rotation = random.random()*2*np.pi-np.pi

            # Generate cluster
            rr, cc = ellipse(r,c,r_radius,c_radius,shape=self.shape,rotation=rotation)

            self.terrain[rr,cc] = i%num_classes
        
        # Update layers list
        self.layers.add('terrain')
        
        return self.terrain

    # def add_layer(self, name, data):
    #     """Add a terrain layer to the current site

    #     Args:
    #         name (str): layer name (will be an attribute of the site)
    #         data (np.array): 2D float numpy array with the same shape
    #             as the other maps.
    #     """

    #     if not self.shape == data.shape:
    #         log.warn(f"Wrong shape: {data.shape}, was expecting {self.shape}")
    #         return
        
    #     if name in self.layers:
    #         log.warn(f"Layer {name} already exists! Choose a different name")
    #         return
        
    #     self.layers.add(name)
    #     setattr(self, name, data)

    # def remove_layer(self, name):
    #     """Remove a layer by its name. Ignore if the layer does not exist
        
    #     Args:
    #         name (str): layer name
    #     """

    #     if name in self.layers:
    #         delattr(self, name)
    #         self.layers.remove(name)
        
    #     return


    @property
    def plotting_extent(self):
        """Return plotting extent for matplotlib."""
        return self._plotting_extent

    def save_dataset(self, save_dir: Path, misc_info: str=''):
        """Save the terrain maps

            Args:
                save_dir: complete path to the dataset directory to create
                misc_info (str, optional): Additional information to add to the
                    configuration metadata like the terrain generation params.

            Returns:
                str or int: Complete path to save subdirectory if succeeded or
                -1 if saving failed.
            
            Notes:
                Maps are saved as .tif files and a configuration metadata .json
                file is also created.
        """

        save_dir.mkdir()
        
        # Save all site layers
        for name in self.layers:

            data = getattr(self, name)

            # Save as a geotiff file
            with rasterio.open(
                Path(save_dir, f'{name}.tif'),
                'w',
                driver='GTiff',
                height=self.shape[0],
                width=self.shape[1],
                count=1,
                dtype=data.dtype,
                crs=None,
                transform=self.map_transform
            ) as dst:
                dst.write(data, 1)
        
        # Create configuration metadata and save it
        # Since this is a synthetic terrain, longitude and latitude are made up
        config_dict = {
            'center_latitude' : None,
            'center_longitude' : None,
            'reference_body': None,
            'layers' : {
                'dem': {'fpath': 'dem.tif', 'min_val': None, 'max_val': None},
                'slope': {'fpath': 'slope.tif', 'units': 'deg', 'max_val': None},
                'aspect': {'fpath': 'aspect.tif', 'units': 'deg'},
            },
            'nogo_processing': {
                'inflation_kernel_dim': None,
                'free_island_is_nogo': False,
                'islands_corners_connect': False,
                'border_is_nogo': True
            }
        }

        if 'terrain' in self.layers:
            config_dict['layers']['terrain'] = {'fpath': 'terrain.tif'}

        with open(Path(save_dir, 'settings.yaml'), 'w') as f:
            yaml.dump(config_dict, f)

        with open(Path(save_dir, 'data_creation_readme.txt'), mode='w') as f:
            f.write(misc_info)
        
        log.info(f"Files successfully saved to: {save_dir}")

        return

