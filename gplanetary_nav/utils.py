#!/usr/bin/env python

""" 
    Utility functions for the gplanetary_nav package

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""


import logging
import os
import yaml
import json
import numba
from typing import Any, IO, Tuple, List, Union

import networkx as nx
from inspect import getfullargspec
from itertools import product, tee, groupby
from inspect import getfullargspec
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

def enumerated_iter(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))

def pairwise(iterable):
    """Iterate and provide the current and next element.
       [a,b,c,d] -> (a,b), (b,c), (c,d)
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def split_quantity(qty, n):
    """Break a quantity down into n (+/- equal) bins

    Ex: split_quantity(10,3) --> [4,3,3]
    
    Args:
        qty (int): quantity to divide
        n (int): num of bins to divide it into
    
    Return:
        list: list of n quantities (quantity per bin)
    """

    q, m = divmod(qty, n)
    breakdown = [q]*n
    for i in range(m):
        breakdown[i] += 1
    return breakdown

def split_array(arr, n):
    """Split an array into similar chunks along axis 0
    
    Args:
        arr (np.ndarray): array
        n (int): number of chunks to break the array into
    
    Returns:
        list: list of array chunks
    """

    qty_split = split_quantity(len(arr),n)

    arr_list = []
    prev_idx = 0
    for bin_size in qty_split:
        arr_list.append(arr[prev_idx:prev_idx+bin_size])
        prev_idx = prev_idx+bin_size
    
    return arr_list

def remove_consecutive_duplicates(iterable: Union[list,tuple]) -> List:
    """Given an (ordered) list or tuple, return a list containing no
    consecutive duplicates. Credit: https://stackoverflow.com/a/5738933

    Ex: (1,2,2,3,1,1,3) -> [1,2,3,1,3]

    Args:
        iterable: list or tuple containing consecutive duplicates
    
    Return:
        list without consecutive duplicates
    """
    return [key for key, _group in groupby(iterable)]


def init_args_to_attributes(values):
    """Set a class' init arguments as class attributes

    Credit: https://stackoverflow.com/a/15484172
    """

    for i in getfullargspec(values['self'].__init__).args[1:]:
        setattr(values['self'], i, values[i])

def arr_totuple(arr: np.array) -> Tuple[tuple]:
    """Convert a NxM numpy array to a N-tuple of M-tuples
    Ex: array [[1,2,3],[4,5,6]] -> ((1,2,3), (4,5,6))
    """
    return tuple(zip(*[arr[:,i] for i in range(arr.shape[1])]))

def bresenham_line(start, end):
    """Bresenham's Line Algorithm, produces a list of tuples from start and end
    Retrieved from https://newbedev.com/python-bresenham-s-line-drawing-algorithm-python-code-example
    
    Args:
        start (tuple): integer grid coordinates of beginning of line
        end (tuple): integer grid coordinates of end of line
    
    Return:
        list: list of grid coordinates from start to end, inclusively
    """
    
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def remove_islands(old_obs: np.array, corner_allowed: bool=False) -> None:
        """Find all separated clusters of free pixels, only keep the largest
           one and set all the other ones as no-go
        
        Args:
            old_obs: the input obstacle binary map (integer-valued, 0 = free,
                1 = obstacle)
            corner_allowed: whether two free clusters only touching
                via a pixel corner (i.e. in 'diagonal') should be considered
                part of the same cluster. Default is False.
        """

        # Map of free space (1=free, 0=obstacle)
        old_free = np.invert(old_obs.astype(bool))

        # Create labelled array
        if corner_allowed:
            s = np.ones((3,3))
        else:
            s = np.zeros((3,3))
            s[1,:] = np.ones(3)
            s[:,1] = np.ones(3)
        labelled_array, num_labels = ndimage.label(old_free, structure=s)

        # Find label with the highest number of elements
        best_label = 1
        size_best_label = len(np.argwhere(labelled_array == best_label))

        for label in range(num_labels)[1:]: # Label 0 is the no-go region

            if len(np.argwhere(labelled_array == label)) > size_best_label:
                best_label = label
                size_best_label = len(np.argwhere(labelled_array == label))
        
        # Update no-go map
        new_free = np.where(labelled_array == best_label, 1, 0).astype(np.bool)
        return np.invert(new_free).astype(old_obs.dtype)


def colorline(x, y, ax, cmap='jet', **plot_kwargs):
    """ Plot a 2D trajectory where segments are colored according to a colormap
    Inspired by this answer: https://stackoverflow.com/a/25941474
    """

    cmap = plt.get_cmap(cmap)
    norm=plt.Normalize(0.0, 1.0)

    # Default colors equally spaced on [0,1]:
    z = np.linspace(0.0, 1.0, len(x))

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              **plot_kwargs)

    ax.add_collection(lc)

    return ax


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def path_cost(G: nx.Graph, path: list, weight: str) -> float:
    """Evaluate the total (edge) cost of a path through a (weighted) graph
    
    Args:
        G: the (weighted) directed or undirected graph
        path: list of sequential nodes in G
        weight: the attribute of edges to use as weights
    
    Return:
        the path cost
    """

    if len(path) == 1:
        return 0

    cost = 0
    for e in pairwise(path):
        cost += G[e[0]][e[1]][weight]
    
    return cost

def masked_argmin(masked_arr: np.ma.array, axis: int=None) -> np.ma.masked_array:
    """argmin operation on a masked array. If the items along a specified
    axis are all masked, the argmin result is np.nan for this sequence.

    Args:
        masked_array: the input masked array
        axis: the axis along which to take the argmax
    Return:
        masked array
    """
    
    idxs = masked_arr.argmin(axis=axis)
    masked = np.all(masked_arr.mask, axis=axis)
    return np.ma.masked_array(data=idxs, mask=masked)

def masked_argmax(masked_arr: np.ma.array, axis: int=None) -> np.array:
    """argmax operation on a masked array, similar to masked_argmin above
    """
    
    idxs = masked_arr.argmax(axis=axis)
    masked = np.all(masked_arr.mask, axis=axis)
    return np.ma.masked_array(data=idxs, mask=masked)


@numba.jit(
    numba.types.int64(numba.types.float64[:], numba.types.float64),
    nopython=True)
def opt_searchsorted_left(arr, val):
    return np.searchsorted(arr, val, side='left')

@numba.jit(
    numba.types.int64(numba.types.float64[:], numba.types.float64),
    nopython=True)
def opt_searchsorted_right(arr, val):
    return np.searchsorted(arr, val, side='right')-1

# @numba.jit(
#     numba.types.int64(numba.types.int64[:], numba.types.int64),
#     nopython=True)
# def opt_searchsorted_int(arr, val):
#     return np.searchsorted(arr, val, side='right')-1

@numba.jit(
    numba.types.int64(numba.types.float64[:], numba.types.float64),
    nopython=True)
def opt_nearest_1d(arr, val):
    return (np.abs(arr-val)).argmin()

# Retrieved from https://stackoverflow.com/a/34325723/18386585
# usage: 
# for item in progressBar(items, ...):
#   do_stuff
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


# Copyright (c) 2018 Josh Bode
# Adapted from https://gist.github.com/joshbode/569627ced3076931b02f
class IncludeLoader(yaml.FullLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: IncludeLoader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, IncludeLoader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def load_yaml_with_includes(fpath):
    """Load a yaml file that includes other yaml files using !include
    commands"""
    yaml.add_constructor('!include', construct_include, IncludeLoader)
    with open(fpath) as f:
        data = yaml.load(f, Loader=IncludeLoader)
    return data
