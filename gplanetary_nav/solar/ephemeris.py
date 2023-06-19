#!/usr/bin/env python

""" 
    Ephemeris model, to retrieve apparent azimuth/elevation of celestial bodies
    using the JPL Horizons server

    Additional documentation:
    https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import logging
import time
from typing import List
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt

import jdcal
from astroquery.jplhorizons import Horizons

from gplanetary_nav.utils import split_array
from gplanetary_nav.solar.utils import SYNODIC_DAY_SECONDS_FROM_NAME, \
    ransac_fit_sin_phase, find_zeros_between


log = logging.getLogger(__name__)


# IDs of different bodies
# (to avoid name confusions, it's always better to specify bodies by their ID)
ID_FROM_NAME = {
    'sun': 10,
    'moon': 301,
    'earth': 399,
    'mars': 499
}

# Max number of timestamps to query at a time on the JPL Horizons server
MAX_QUERY_SIZE = 100

class Ephemeris:

    def __init__(self, lat: float, lon: float, elevation: float, body_name: str):
        """Set the location of the observer

        Args:
            lat: latitude of observer w.r.t. equator (+ve: Northward,
                -ve: Southward), decimal degrees
            lon: longitude of observer (+ve Eastward), decimal degrees
            elevation: elevation above reference ellipsoid (km)
            body_name: name of celestial body where observer is located
        """

        self.lat = lat
        self.lon = lon
        self.body_name = body_name.lower()

        try:
            self.location = {
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'body': ID_FROM_NAME[self.body_name]
            }
        except KeyError as e:
            log.error(f"Wrong body name ({body_name}), one of {ID_FROM_NAME.keys()} expected")
            return

    def az_el_from_timestamps(
        self, utc_timestamps: list, target_name: str='sun') -> pd.DataFrame:
        """Azimuth & elevation angles from a list of UTC unix timestamps

        Args:
            utc_timestamps: list of UTC linux timestamps (seconds)
            target_name: name of celestial body to observe
        Return:
            a dataframe with timestamps, azimuth (deg) and elevation (deg) data
        """

        # Convert timestamps to UTC julian dates
        julian_dates_utc = []
        for timestamp in utc_timestamps:
            dt = datetime.utcfromtimestamp(timestamp)
            jd_utc = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
            
            if self.body_name == 'mars':
                # add in hours & minutes (greater time resolution)
                jd_utc += (dt.hour*3600+dt.minute*60)/(24*3600)
                # jd_utc += (dt.hour*3600+dt.minute*60+dt.second)/(24*3600) # this fails
            julian_dates_utc.append(round(jd_utc, 2))  # rounding up to >2 decimals fails
        
        # Break down into manageable sizes for the Horizons servers
        jd_queries = split_array(
            julian_dates_utc, math.ceil(len(julian_dates_utc)/MAX_QUERY_SIZE))
        
        query_results = []
        for jd in jd_queries:
            
            success_call = False
            while not success_call:
                try:
                    obj = Horizons(
                            id=ID_FROM_NAME[target_name.lower()],
                            location=self.location,
                            epochs=jd)

                    query_results.append(obj.ephemerides())
                    success_call = True
                except ValueError:
                    log.warn(f"Error in server call, trying again")
                    time.sleep(2.0)
                    continue  
        
        return self.extract_results(query_results)
    
    def extract_results(self, ephe_tables: list) -> pd.DataFrame:
        """Extract date, time, elevation and azimuth data from a sequence of
            raw results & combine them into a single dataframe
        
        Args:
            ephe_tables: list of ephemeris result tables
        Return:
            a dataframe with timestamps, azimuth (deg) and elevation (deg) data
        """

        dataframes = []

        for ephe_table in ephe_tables:
            results = {
                'datetime_str' : [],
                'unix_timestamp_s': [],
                'azimuth_deg': [],
                'elevation_deg': []
            }

            for row in ephe_table:

                # ignore decimal seconds, if any
                dt_string = row['datetime_str'].split('.')[0]

                dt = datetime.strptime(dt_string, '%Y-%b-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                results['datetime_str'].append(dt_string)
                results['unix_timestamp_s'].append(dt.timestamp())
                results['azimuth_deg'].append(row['AZ'])
                results['elevation_deg'].append(row['EL'])
            
            dataframes.append(pd.DataFrame(results))
        
        if len(dataframes) == 1:
            return dataframes[0]
        else:
            return pd.concat(dataframes, ignore_index=True, axis=0)
    
    def solar_elevation_extrema(
        self, start_timestamp: float, end_timestamp: float,
        extrema_type: str='minima', resolution: float=36000,
        n_ransac_frac: float=0.75, threshold_error: float=0.1,
        threshold_inlier_frac: float=0.5) -> List[float]:
        """Timestamps of the local extrema (minima or maxima) of the Sun's 
            elevation profile (i.e. 'midnight' or 'noon' timestamps) from the
            observer's perspective. If no such extrena exists over the time
            interval requested, an empty list is returned.

            These timestamps are approximate, based on the time resolution
            requested.

        Args:
            start_timestamp: start of time interval UTC linux timestamp (seconds)
            end_timestamp: end of time interval UTC linux timestamp (seconds)
            extrema_type: 'minima' (for solar 'midnights') or 'maxima' (for
                solar 'noons').
            n_ransac_frac: fraction of the sinusoid vector entries to sample
                during ransac fitting
            threshold_error: the maximum difference between the ransac fitted
                function and a data point for it to be considered an inlier
            threshold_inlier_frac: the fraction of inliers among all data points
                for the fitted function to be considered as acceptable.
            resolution: resolution (seconds) of the midnight timestamps. 
                (Lower = better). Use with caution; the number of requested
                timestamps from the JPL Horizons server is on the order of
                (end_timestamp-start_timestamp)/resolution.
        
        Return:
            timestamps of solar elevation extrema
        """

        # Pad time interval by 1 synodic day on each side of requested interval
        req_start = start_timestamp - SYNODIC_DAY_SECONDS_FROM_NAME[self.body_name]
        req_end = end_timestamp + SYNODIC_DAY_SECONDS_FROM_NAME[self.body_name]

        n_samples = int((req_end-req_start)/resolution)
        timestamps = np.linspace(req_start, req_end, n_samples)

        # Query JPL Horizons server
        ephe_df = self.az_el_from_timestamps(timestamps, target_name='sun')

        # Numerical derivative of elevation profile
        # Solar elevation profiles aren't constant (peaks/valleys of a day
        # vary over time).
        # The derivative of the elevation profile is a more constant sinusoid
        dx = np.average(np.diff(timestamps))
        d_elev = np.correlate(ephe_df['elevation_deg'], [-1,0,1], mode='same')/(2*dx)

        ####################
        # LOW-PASS FILTERING
        ####################

        # Low-pass filter (butterworth filter) on the elevation derivative
        

        # FILTER PARAMETERS
        fs = n_samples/(req_end-req_start)  # sampling frequency
        fnyq = 0.5*fs                       # nyquist frequency

        # (exclude boundaries, which are outliers)
        buffer = int(0.5*SYNODIC_DAY_SECONDS_FROM_NAME[self.body_name]*fs)
        t_filter = timestamps[buffer:-buffer]
        data_filter = d_elev[buffer:-buffer]

        # Desired cutoff frequency
        # (*1.5 to have a cutoff slightly higher than the actual)
        fc = 1/SYNODIC_DAY_SECONDS_FROM_NAME[self.body_name]*1.5

        # Normalized frequency
        fc_norm = fc/fnyq

        # Order (2 is good enough for a sinusoidal signal)
        order = 2

        b, a = butter(order, fc_norm, btype='low', analog=False)
        d_elev_smooth = filtfilt(b, a, data_filter)

        # low-pass filter creates artifacts near the signal boundaries, ignore
        buffer = int(0.5*SYNODIC_DAY_SECONDS_FROM_NAME[self.body_name]*fs)
        d_elev_smooth = d_elev_smooth[buffer:-buffer]
        t_filter = t_filter[buffer:-buffer]

        # Normalize the amplitude & vertically center the smoothed signal
        amp = 0.5*(np.max(d_elev_smooth)-np.min(d_elev_smooth))
        center = np.min(d_elev_smooth)+amp
        d_elev_smooth_norm = (d_elev_smooth-center)/amp

        #####################
        # RANSAC SINUSOID FIT
        #####################
        best_results = ransac_fit_sin_phase(t_filter, d_elev_smooth_norm,
            n=int(n_ransac_frac*len(d_elev_smooth_norm)),
            threshold_error=threshold_error,
            threshold_inlier=int(threshold_inlier_frac*len(d_elev_smooth_norm)),
            w=2*np.pi/SYNODIC_DAY_SECONDS_FROM_NAME[self.body_name],
            c=0,
            A=1.0)

        zeros = find_zeros_between(best_results, req_start, req_end)

        # Retrieve minima or maxima
        extrema = []
        for z in zeros:
            if extrema_type == 'minima':
                if best_results['fitfunc'](z-best_results['period']/4)-best_results['fitfunc'](z) < 0:
                    extrema.append(z)
            elif extrema_type == 'maxima':
                if best_results['fitfunc'](z-best_results['period']/4)-best_results['fitfunc'](z) > 0:
                    extrema.append(z)
            else:
                raise ValueError(
                    f"extrema type unrecognized ({extrema_type}), must be "
                    f"'minima' or 'maxima'")
        
        # Only keep extrema in the time interval initially provided
        valid_extrema = [e for e in extrema if start_timestamp <= e <= end_timestamp]
        return valid_extrema









