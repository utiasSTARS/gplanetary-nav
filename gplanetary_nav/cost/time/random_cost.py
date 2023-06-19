#!/usr/bin/env python

""" 
    Random smooth cost function using Perlin noise

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import os
import sys
import logging
import copy
import math
import random
import time as pytime
from inspect import getfullargspec

import numpy as np
import pickle as pkl
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import MinMaxScaler

from gplanetary_nav.external.perlin_noise_factory import PerlinNoiseFactory
from gplanetary_nav.utils import pairwise, init_args_to_attributes

log = logging.getLogger(__name__)


class PerlinCost:
    """ Create a positive-valued random cost function based on
        Perlin noise
    """

    def __init__(self, extent, rough=1, desired_range=(0,100), res=10):
        """ Initialize random function

        Args:
            extent (list): [(feat0_min,feat0_max), (feat1_min,feat1_max), ...]
                Minimum and maximum value of each feature along each cost axis.
            roughness (int): octaves of the perlin noise, representing
            the roughness of the cost function.
            desired_range (float, float): range of the desired cost function
                over the normalized range of feature values ([0...1])
            res (int): resolution of the cost grid, affecting the actual cost
                range accuracy & plotting resolution. A high res value with 
                high d will take a very long time.
        """

        # Turn every init argument intro an attribute to self
        init_args_to_attributes(vars())

        # Data scaler
        self.scaler = MinMaxScaler()
        mins = [feat[0] for feat in extent]
        maxs = [feat[1] for feat in extent]
        self.scaler.fit([mins, maxs])

        # Sample perlin noise once
        self.d = len(self.extent)
        self.pnf = PerlinNoiseFactory(self.d, octaves=rough, unbias=False)

        # Eval noise over normalized feature space
        feature_space = [np.linspace(0,1,self.res)]*self.d
        mgrids = np.meshgrid(*feature_space)
        
        # Get simulated data matrix (sparsely) covering the feature space
        flat_feature_space = [dd.ravel() for dd in mgrids]
        X = np.array(list(zip(*flat_feature_space)))

        # Cost function
        self.raw_cost = np.zeros(X.shape[0])
        for i, data in enumerate(X):
            self.raw_cost[i] = self.pnf(*data)
        
        self.raw_cost_range = (self.raw_cost.min(), self.raw_cost.max())
        self.cost = np.interp(self.raw_cost, self.raw_cost_range,
                              self.desired_range)
        self.cost = self.cost.reshape(mgrids[0].shape, order='F')

    def eval(self, X):
        """Evaluate the cost on every row of the provided data matrix
        
        Args:
            X (np.array): (n,d) array, where each row contains a data entry and
                d is the number of features per entry.
        
        Return:
            np.array: (n,) array of costs for the data matrix provided
        """

        # Ensure 2D array provided
        if len(X.shape) == 1:
            X = X.reshape((1,-1))
        
        X_scaled = self.scaler.transform(X)
        log.info(f"X_scaled: {X_scaled}")

        # Sanity check
        if X_scaled.shape[1] != self.d:
            log.error(f"Wrong number of features in data matrix: {X_scaled.shape[1]} "
                      f"(was expecting {self.d})")
            return

        # Get raw costs
        raw_cost = np.zeros(X_scaled.shape[0])
        for i, data in enumerate(X):
            log.debug(f"Evaluating data {data}")
            raw_cost[i] = self.pnf(*data)
        
        # Scale raw costs to get costs in desired range
        cost = np.interp(raw_cost, self.raw_cost_range, self.desired_range)
        return cost

    def plot_2d(self, ax=None, plot_features=[0,1], other_idx=0):
        """ Slice the 2D cost array along the specified axes & plot it
        
        Args:
            plot_features [int,int]: axes (features) along which the 2D cost
                array should be extracted
            other_idx (int): index of the other (non-extracted) axes where the 
                slice should be taken.
        """

        indexing = [slice(None) if i in plot_features else other_idx for i in range(self.d)]

        ret_ax = ax is not None
        if ret_ax:
            fig = plt.Figure()  # dummy figure object for colorbar function
        else:
            fig, ax = plt.subplots()  # create own ax object
            
        
        # Make room for colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(self.cost[tuple(indexing)], extent=[self.extent[1][0],self.extent[1][1],self.extent[0][0],self.extent[0][1]])
        ax.set_aspect('equal', 'box')
        fig.colorbar(im, cax=cax, orientation='vertical')

        # Title & axes labelling
        indexing = [None if i in plot_features else other_idx for i in range(self.d)]
        indexing_str = str(indexing).replace('None', ':')
        ax.set_title(f"Cost function slice from cost{indexing_str}")

        # Highest axis index along hor. axis
        ax.set_xlabel(f"Normalized feature {max(plot_features)}") 
        ax.set_ylabel(f"Normalized feature {min(plot_features)}")

        if ret_ax:
            return ax
        else:
            plt.show()
    
    @property
    def data_matrix(self):
        """(num edges, 1 + num features) data matrix
        The first column is a categorical feature (all zeros in this case)
        """

        data_matrix = np.hstack((np.zeros((self.X.shape[0],1)), self.X))
        return data_matrix
    
    def __str__(self):
        s = str(f"\nRandom Perlin cost with roughness (octaves) {self.rough} "
            +f"with dimensionality {self.d} "
            +f"and true edge cost between {self.desired_range}, "
            +f"edge cost multiplied by edge length: {self.incl_len}"
            +f"\nGraph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
        return s