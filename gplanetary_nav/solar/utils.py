#!/usr/bin/env python

""" 
    Utilities for the solar module

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from datetime import timedelta
import scipy.optimize
import numpy as np
import math
import random


# Average synodic day lengths, in seconds
SYNODIC_DAY_SECONDS_FROM_NAME = {
    'moon': timedelta(days=29, hours=12, minutes=44, seconds=3).total_seconds(),
    'earth': timedelta(days=1).total_seconds(),
    'mars': timedelta(days=1, minutes=39, seconds=35).total_seconds(),
}

# Average solar constants (W/m^2) from body name
SOLAR_CONSTANTS_FROM_NAME = {
    'moon': 1367,
    'earth': 1367,
    'mars': 589,
}

def fname_from_coord(coords: tuple, ext: str='csv') -> str:
    """Return the solar flux interval .csv file name of a given pixel
    
    Args:
        coords ((int,int)): tuple of (row,col) pixel coordinates
    
    Return:
        str: return the complete .{ext} file name of that pixel
    """
    
    return f"{coords[0]:05d}-{coords[1]:05d}.{ext}"


def ransac_fit_sin_phase(
    tt: np.array, yy: np.array, n: int, threshold_error: float,
    threshold_inlier: int, w: float=1.0, c: float=0, A:float=1.0,
    max_iters: int=100) -> dict:
    """ RANSAC for fitting a sine function phase parameter

    Args:
        tt: (n,) array of time component of the data
        yy: (n,) array of sinusoid component of data points
        n: number of points to sample at each iteration
        threshold_error: max (vertical) distance from a point to the fitted 
            curve to be considered an inlier
        threshold_inlier: miminum number of inlier points to consider a model
        w, c, A: parameters of the sin function y(t) = A*sin(w*t+p) + c
            (p is the phase parameter to fit)
        max_iters: the maximum number of RANSAC iterations
    
    Return:
        fit_sin() return dictionary of the best fit
    """

    k = 0
    best_score = np.inf
    best_results = None
    while k < max_iters:

        indices = sorted(random.sample(list(range(len(tt))), k=n))

        try:
            guess = np.array([0])   # initial guess for p
            def sinfunc(t, p): return A*np.sin(w*t+p)+c
            popt, pcov = scipy.optimize.curve_fit(
                sinfunc, tt[indices], yy[indices], p0=guess)

            p = popt[0]
            f = w/(2.*np.pi)
            fitfunc = lambda t: A*np.sin(w*t + p)+c
            results = {
                "amp": A, "omega": w, "phase": p, "offset": c, "freq": f,
                "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov),
                "rawres": (guess,popt,pcov)
            }
            
            # results = fit_sin(tt[indices], yy[indices])
        except RuntimeError:
            k+=1
            continue

        distances = np.abs(yy-results['fitfunc'](tt))
        # Inliers
        inlier_idxs = np.argwhere(distances < threshold_error).flatten()
        log.info(f"Inlier count: {len(inlier_idxs)} (threshold: {threshold_inlier})")
        if len(inlier_idxs) > threshold_inlier:
            
            # Average error among inliers
            score = np.average(distances[inlier_idxs])
            if score < best_score:
                best_score = score
                best_results = results
            
                log.info(f"Inliers: {len(inlier_idxs)}, Score: {score}")

        k+=1
    
    if best_results is None:
        raise ValueError("No good fit found, adjust thresholds")

    return best_results


def find_zeros_between(results, lower: float, upper: float) -> list:
    """Find function zeros between a lower and an upper bound
    
    Args:
        results: results object returned by ransac_fit_sin
        lower: lowest value for the zero
        upper: highest value for the zero

    Result:
        zeros within the specified range
    """

    zeros_in_range = []

    # Zero formula given an integer k
    zero_f = lambda k: (np.pi*k-results['phase'])/results['omega']

    smallest_k = math.ceil((lower*results['omega']+results['phase'])/np.pi)

    k = smallest_k
    zero = zero_f(k)
    while lower <= zero <= upper:
        zeros_in_range.append(zero)
        k+=1
        zero = zero_f(k)
    
    return zeros_in_range