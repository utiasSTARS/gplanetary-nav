#!/usr/bin/env python

""" 
    Solar power model loader

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import copy
import functools
import os
from pathlib import Path
import logging
import csv
import json
from typing import Union, Tuple, Set, List
from datetime import datetime, timedelta

import rasterio
import yaml
import numpy as np
import numba
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

# from gplanetary_nav.solar.irradiance_mapping import IrradianceMappingPercentage
from gplanetary_nav.solar.utils import fname_from_coord, \
    SYNODIC_DAY_SECONDS_FROM_NAME, SOLAR_CONSTANTS_FROM_NAME
from gplanetary_nav.planning.types import Node, NodeStamped
from gplanetary_nav.solar.numba_energy import NumbaEnergy, node_power_numba_d
from gplanetary_nav.utils import load_yaml_with_includes


log = logging.getLogger(__name__)

def load(dirpath: str, rover_cfg: dict=None) -> object:
    """Load solar power & irradiance models

    Args:
        dirpath: complete path to the dataset directory
        rover_cfg: rover specification dictionary. If None, a dummy
            rover model is loaded (useful if only basic SolarLoader utilities
            like irradiance map access are needed).

    Returns:
        SolarLoader : solar loader instance of the requested site
    """

    if rover_cfg is None:
        # Load a dummy rover model
        from gplanetary_nav.dev.models import DUMMY_ROVER_CFG as rover_cfg
        log.warning(f"No rover config provided, using dummy one: \n{rover_cfg}")

    # Check that the solar data products exist
    irr_config_dir = Path(dirpath, "irradiance")
    if not irr_config_dir.exists():
        raise FileNotFoundError(f"Irradiance directory not found: {irr_config_dir}")
    
    return SolarLoader(irr_config_dir, rover_cfg)

class SolarLoader:
    def __init__(self, irr_config_dir: Path, rover_cfg: dict) -> None:
        """Load requested configurations
        
        Args:
            irr_config_dir: complete path to the irradiance directory
            rover_cfg: rover specification
        """

        # Load rover specifications
        
        # Area in m^2
        self.area = rover_cfg['solar_panel']['area']

        # Efficiency as float from 0 (0%) to 1 (100%)
        self.efficiency = rover_cfg['solar_panel']['efficiency']

        # Load irradiance paths
        self.irr_config_dir = irr_config_dir
        self.irr_maps_dir = Path(irr_config_dir, 'maps')
        self.px_interv_dir = Path(self.irr_config_dir, 'pixel_intervals')

        # Load reference body name
        settings_fpath = Path(self.irr_config_dir.parent, 'settings.yaml')
        settings = load_yaml_with_includes(settings_fpath)
        self.reference_body = settings['reference_body'].lower()
        
        # Retrieve valid time intervals and irradiance filenames of model
        self.times_fnames = dict()
        fpath = Path(self.irr_config_dir, "times.csv")
        with open(fpath, mode='r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                self.times_fnames[int(row["unix_timestamp_s"])] = row["fname"]
        
        sorted_times = sorted(self.times_fnames.keys())
        self.time_resolution = sorted_times[1]-sorted_times[0]

        # Flag indicating whether power timeseries were pre-loaded
        self.all_timeseries_loaded = False
    
    def load_all_timeseries(self, node_list: List[Node]=None) -> None:
        """Load all power timeseries (at every node) in memory

        Args:
            node_list: list of tuple grid coordinates reachable by the rover
        """

        if node_list is None:
            # Load all preprocessed locations
            with open(Path(self.px_interv_dir, 'valid_coords.json'), 'r') as f:
                node_list = list(map(tuple, json.load(f)))

        # Re-calculate the maximum power only using the loaded data
        self.max_power = 0

        self.node_power = dict()
        for node in node_list:
            ts = self.power_timeseries(node)
            self.node_power[node] = ts
            node_power_numba_d[node] = ts

            self.max_power = max(self.max_power, np.nanmax(ts[:,1]))
        
        self.numba_energy = NumbaEnergy(node_power_numba_d)

        self.all_timeseries_loaded = True
    
    def irradiance_timeseries(self, node: tuple) -> np.array:
        """Load the irradiance timeseries at a node
        
        Args:
            node: tuple of grid coordinates
        
        Return:
            2D array. col 1 is the timestamps, col 2 is the irradiance in W/m^2
        """

        fpath = Path(self.px_interv_dir, fname_from_coord(node))

        try:
            with open(fpath, mode='r') as f:
                next(f) # skip first row (column headers)
                return np.array(list(csv.reader(f))).astype(np.float32)
        except FileNotFoundError as e:
            raise e(
                f"Irradiance timeseries not found for node {node}. Have you "
                "ran the preprocessing module & generated irradiance intervals?")
    
    def power_timeseries(self, node: Node) -> np.array:
        """Power timeseries at a given node/location
        
        Args:
            node: the location (pixel coords)
        
        Return:
            2D array. col 1 is the timestamps, col 2 is the solar power
                generated (in W), considering solar panel area & efficiency.
        """

        if self.all_timeseries_loaded:
            return self.node_power[node]
        else:
            irr_timeseries = np.copy(self.irradiance_timeseries(node))
            irr_timeseries[:,1] *= self.area*self.efficiency # convert to power
            return irr_timeseries
    
    def power(self, node_stamped: NodeStamped) -> float:
        """Power delivered to the batteries at a specified location & time
        
        Args:
            node_stamped: the location & time requested
        
        Return:
            float: instantaneous solar power generated (W)
        """

        power_timeseries = self.power_timeseries(node_stamped.node)
        time_idx = np.searchsorted(power_timeseries[:,0], node_stamped.time)-1
        return power_timeseries[time_idx,1]

    
    # def energy(self, from_node_stamped: NodeStamped, to_node_stamped: NodeStamped) -> float:
    #     """Energy delivered to the batteries during an action
        
    #     Args:
    #         from_node_stamped: the location & time at the beginning of the
    #             action
    #         to_node_stamped: the location & time at the end of the action. Must
    #             be a location neighbouring the from_node (or the same node)
        
    #     Return:
    #         float: energy delivered to the batteries (J)
    #     """

    #     if to_node_stamped.time-from_node_stamped.time < self.time_resolution:
    #         # For short actions (typically drive actions)

    #         # Use average between both node stamped
    #         from_power = self.power(from_node_stamped)
    #         to_power = self.power(to_node_stamped)
    #         avg_power = (from_power+to_power)/2

    #         dur = to_node_stamped.time - from_node_stamped.time

    #         return avg_power*dur/3600  # Convert duration from s to hr
        
    #     else:
    #         # Integration method. This is only for long actions
    #         # (ONLY FOR LONG WAIT ACTIONS - assumes no node change)
    #         assert from_node_stamped.node == to_node_stamped.node

    #         power_intervals = self.power(from_node_stamped, ret_series=True)

    #         # log.info(f"Irradiance intervals: {irr_intervals}, from_node time: {from_node_stamped.time}, to_node time: {to_node_stamped.time}")
    #         # log.info(f"Type: {type(irr_intervals)}")


    #         idx_min = np.searchsorted(power_intervals[:,0], from_node_stamped.time)-1
    #         idx_max = np.searchsorted(power_intervals[:,0], to_node_stamped.time)-1

    #         energy = 0
    #         for idx in np.arange(idx_min, idx_max+1):
    #             power = power_intervals[idx,1]
    #             if idx == idx_min:
    #                 dur = power_intervals[idx+1,0] - from_node_stamped.time
    #             elif idx == idx_max:
    #                 dur = to_node_stamped.time - power_intervals[idx,0]
    #             else:
    #                 dur = power_intervals[idx+1,0]-power_intervals[idx,0] #self.time_resolution

    #             energy += power*dur/3600  # Convert duration from s to hr
            
    #         return energy
    
    def energy(self, node: Node, t_bounds: Tuple[float]) -> float:
        """Energy delivered to the batteries at a location over a time period
            (Numba-optimized!)

        To use the numba-optimized utility, call self.load_all_timeseries()
        beforehand.

        This only considers the insolation conditions in one location/node. For
        a drive action going from a start node to a (different) goal node, call
        this method twice (with the appropriate node & time bounds for each).
        
        Args:
            node: the location (pixel coords) in which the action is taken
            t_bounds: (start time, end time) bounds, each a unix timestamp (s)
        
        Return:
            float: energy delivered to the batteries (Whr)
        """

        if self.all_timeseries_loaded:
            return self.numba_energy.energy(node, t_bounds)
        else:
            # log.warning(
            #     f"The load_all_timeseries() was not called, using "
            #     f"non-optimized solar energy calculation")
            power_intervals = self.power_timeseries(node)

            idx_min = np.searchsorted(power_intervals[:,0], t_bounds[0])-1
            idx_max = np.searchsorted(power_intervals[:,0], t_bounds[1])-1

            if idx_min == idx_max:
                # (/3600 to convert duration from s to hr)
                return power_intervals[idx_min,1]*(t_bounds[1]-t_bounds[0])/3600

            energy = 0
            energy += (power_intervals[idx_min+1,0]-t_bounds[0])*power_intervals[idx_min,1]
            energy += (t_bounds[1]-power_intervals[idx_max,0])*power_intervals[idx_max,1]

            if idx_min+1 == idx_max:
                return energy/3600 # convert W-s to W-hr
            else:
                energy += np.dot(
                    np.diff(power_intervals[idx_min+1:idx_max+1,0]),
                    power_intervals[idx_min+1:idx_max,1])

                return energy/3600 # convert W-s to W-hr

    def irradiance_intervals(self, coords):
        """ Return the irradiance intervals at a specific location

        Args:
            coords ((int,int)): tuple of map pixel coordinates

        Return:
            np.ndarray: (n,2) array where n is the number of intervals. The
                first column is the unix timestamp (s) and the second is
                the irradiance in W/m^2 
        """

        fpath = os.path.join(self.px_interv_dir, fname_from_coord(coords))
        with open(fpath, mode='r') as f:
            next(f) # skip first row (column headers)
            irr_intervals = np.array(list(csv.reader(f))).astype(np.float32)
        return irr_intervals

    def sun_angles(self, node_stamped: NodeStamped) -> Union[float,float]:
        """Sun azimuth and elevation angles at the specified location & time
        
        Args:
            node_stamped: the location & time requested
        
        Return:
            (float,float): azimuth from North (deg, orientation convention?) and
                elevation above the horizon (deg)
        """

        pass

    def map_at_timestamp(self, timestamp: float, ret_extent: bool=False):
        """Irradiance map corresponding to the provided timestamp

        Invalid data is set to an irradiance of 0

        Args:
            timestamp: map applicable at this unix timestamp (s)
            ret_extent: whether to return the plotting extent of the map
                Default: False
        
        Return:
            np.array: the irradiance map (units: W/m^2)
            list: map plotting extent, if ret_extent is True
        """

        # Find irradiance map key (timestamp) preceding the provided stamp
        idx = np.searchsorted(sorted(self.times_fnames.keys()), timestamp, side='right')-1
        # idx = max(idx-1,0)
        time_key = sorted(self.times_fnames.keys())[idx]

        fname = self.times_fnames[time_key]
        dataset = rasterio.open(Path(self.irr_maps_dir, fname))
        arr = dataset.read(1)

        # Invalid pixels assumed to be in the shade
        arr[arr==dataset.meta['nodata']] = 0
        arr = np.nan_to_num(arr, nan=0)

        # Clip values to valid intervals
        arr = np.clip(arr, a_min=0, a_max=self.max_irradiance)

        if ret_extent:
            return (arr, rasterio.plot.plotting_extent(dataset))
        else:
            return arr
    
    def plot_irradiance_map(self, ax: Axes, timestamp: float, cmap: str='jet', cbar=True) -> Axes:
        """Plot the average solar irradiance map in a given time window
        
        Args:
            ax: matplotlib Axes instance
            timestamp: map applicable at this unix timestamp (s)
            cmap: the matplotlib colormap to use
            cbar: whether to show the color
        
        Return:
            updated matplotlib Axes instance
        """

        irr_map = self.map_at_timestamp(timestamp)
        date = datetime.utcfromtimestamp(timestamp)

        im = ax.imshow(irr_map, extent=self.extent, cmap=cmap,
            vmin=0, vmax=self.max_irradiance)

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)

        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")
        ax.set_title(
            f"Irradiance map (W/m^2) for {date}\n"
            f"Invalid data set to 0.")

        return ax

    @property
    def dates(self):
        """Datetime of all irradiance maps available"""
        return list(map(datetime.utcfromtimestamp, self.times))

    @property
    def times(self):
        """Unix timestamps of all irradiance maps available"""
        return list(self.times_fnames.keys())

    @property
    def num_maps(self):
        """Number of irradiance maps in the dataset"""
        return len(self.times)
    
    @property
    def avg_time_window(self):
        """Average time window duration (s) over which an irradiance map applies"""
        return np.average(np.diff(self.times))

    @property
    def min_time(self):
        return self.times[0]
    
    @property
    def max_time(self):
        return self.times[-1] + self.avg_time_window
    
    @property
    def min_date(self):
        return datetime.utcfromtimestamp(self.min_time)
    
    @property
    def max_date(self):
        return datetime.utcfromtimestamp(self.max_time)
    
    @property
    def max_irradiance(self):
        """Maximum solar irradiance across preprocessed data, in W/m^2"""
        try:
            with open(Path(self.px_interv_dir, 'max_irradiance.txt'), mode='r') as f:
                max_power = float(f.read())*self.area*self.efficiency
            return max_power
        except FileNotFoundError:
            log.warning(
                f"Max irradiance of preprocessed data not found, "
                f"returning solar constant instead.")
            return SOLAR_CONSTANTS_FROM_NAME[self.reference_body]
    
    @property
    def day_secs(self) -> float:
        """Average length of a synodic ('solar') day on the reference body, in s"""
        return SYNODIC_DAY_SECONDS_FROM_NAME[self.reference_body]
    
    @property
    def extent(self) -> list:
        """Matplotlib plotting extent of solar maps [left,right,bottom,top]"""
        return self.map_at_timestamp(self.times[0], ret_extent=True)[1]
    
    @property
    @functools.lru_cache(maxsize=1) 
    def psr_coords(self) -> Set[Node]:
        """Set of coordinates in PSRs"""
        avg_arr = np.ma.array(self.average_irradiance_map(), mask=self.nogo)
        return set(map(tuple, np.argwhere(avg_arr==0).tolist()))

    @functools.lru_cache(maxsize=1) 
    def average_irradiance_map(
        self, time_start: float=None, time_end: float=None) -> np.array:
        """Average irradiance received based on maps whose associated timestamp
        is between a provided time window

        Args:
            time_start: Unix timestamp of the start of the time window
            time_end: Unix timestamp of the end of the time window
        
        Return:
            array of average irradiance, in W/m^2
        """

        lower = time_start if time_start is not None else -np.inf
        upper = time_end if time_end is not None else np.inf

        # Array summed over, initialized at 0
        summed_arr = 0*self.map_at_timestamp(self.times[0])

        count = 0
        for timestamp in self.times:
            if lower <= timestamp <= upper:
                summed_arr += self.map_at_timestamp(timestamp)
                count += 1
        
        return summed_arr/count  # take the average
    
    def plot_average_irradiance_map(
        self, ax: Axes, time_start: float=None, time_end: float=None, cbar: bool=True, cmap: str="jet") -> Axes:
        """Plot the average solar irradiance map in a given time window
        
        Args:
            ax: matplotlib Axes instance
            time_start: Unix timestamp of the start of the time window
            time_end: Unix timestamp of the end of the time window
            cbar: whether to show the colorbar (default: True)
            cmap: the matplotlib colormap to use (default: 'jet')
        Return:
            updated matplotlib Axes instance
        """

        avg_arr = self.average_irradiance_map(time_start, time_end)
        avg_arr_psr = np.ma.array(avg_arr, mask= avg_arr == 0)
        imcmap = copy.copy(cm.get_cmap(cmap))
        imcmap.set_bad('black')

        im = ax.imshow(avg_arr_psr, extent=self.extent, cmap=imcmap,
            vmin=0, vmax=self.max_irradiance)
        # ax.axis('equal')

        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)

        ax.set_xlabel("Easting (meters)")
        ax.set_ylabel("Northing (meters)")
        ax.set_title(
            f"Average solar irradiance (W/m^2) \n(black = invalid or no irradiance over"
            f" time window specified)")

        return ax
    
    @property
    def nogo(self):
        """Locations where illumination data is unavailable
        This is solely based on where nan values are located on the first
        irradiance map of the dataset.
        """

        time_key = sorted(self.times_fnames.keys())[0]
        fname = self.times_fnames[time_key]
        arr_src = rasterio.open(Path(self.irr_maps_dir, fname))
        arr = arr_src.read(1)

        return np.logical_or(arr<0, np.isnan(arr), arr == arr_src.meta['nodata'])
            

    def create_animation(self, output_fpath):
        """Create an animation of the irradiance maps
        
        Args:
            output_fpath (str): complete path the desired output file
                (including the .mp4 extension)
        """
        
        # Show first irradiance map
        map_data, extent = self.map_at_timestamp(self.times[0], ret_extent=True)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)

        im = ax.imshow(map_data, cmap='gray', vmin=0,
                       vmax=self.max_power, extent=extent)
        fig.colorbar(im, ax=ax)
        
        self.plots = [im]

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_title("Irradiance (W/m^2) over time")

        anim = FuncAnimation(fig, self.animate, frames=len(self.times_fnames),
        interval=150, blit=True)

        anim.save(output_fpath, extra_args=['-vcodec', 'libx264'], dpi=150)

    def animate(self, i):
        map_data = self.map_at_timestamp(self.times[i])
        self.plots[0].set_array(map_data)
        return self.plots

