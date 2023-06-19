#!/usr/bin/env python

""" 
    Smoothly-varying velocity function

    Authors: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import time
import logging

import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class GaussianPDFVelocity:
    """Sample smooth velocity function with the shape of a Gaussian PDF"""
    
    domain_lim = np.array([
            [0,0.15],   # CFA range
            [-25,25]    # Rover pitch (degrees) range
    ])
    terrain_domain = np.linspace(1,5,5)

    def __init__(self):
        pass

    def in_domain(self, X, raise_error=True):
        """Check whether an input is inside the continuous function domain

        Args:
            X (array_like): (2,) array of [cfa,pitch]
                (cfa within [0,1] , pitch in degrees)
            raise_error (bool): if out of bounds, raise a ValueError
        """

        for i, d_x in enumerate(self.domain_lim):
            if (not d_x[0] <= X[i] <= d_x[1]):
                if raise_error:
                    raise ValueError(f"Input along dimension {i} ({X[i]}) outside of function domain")
                return False
        
        # If everything is in bounds
        return True

    def eval(self, features_dict):
        """ Velocity given rover and terrain features
        
        Args:
            features_dict: dictionary of terrain and rover features
                (rover_pitch in rad, cfa within [0,1], terrain (default 1)).

        Return:
            np.ndarray: (n,) array of velocities
        """

        try:
            terrain = features_dict['terrain']
        except KeyError:
            terrain = 1
        
        pitch = np.degrees(features_dict['rover_pitch'])
        cfa = features_dict['cfa']
        X = np.array([cfa, pitch])

        # Location of highest velocity [cfa, pitch] as the Gaussian mean
        mu = np.array([0,-3])

        # Covariance of Gaussian distribution
        # (how quickly velocity drops down to 0 with increasing cfa & pitch)
        C = np.array([
            [0.015,0],
            [0,250]
        ])

        # Sanity check
        if len(X.shape) == 1:
            X = X.reshape((-1,2))

        vals = []
        for x in X:

            if not self.in_domain(x):
                return

            exponent = -0.5*((x-mu).T).dot(np.linalg.inv(C)).dot(x-mu)
            val = (2*np.pi)**(-1)*np.linalg.det(C)**(-0.5)*np.exp(exponent)
            vals.append(val)
        
        # Low terrain values are fast, high terrain values are slow
        # Velocity is 0 for terrain 5
        terrain_multiplier = 200*(5-terrain)

        velocities = np.array(vals)*terrain_multiplier

        # According to Ono (2018), lowest velocity is 10.9 m/hr,
        # which we round down to 10 m/hr. Anything lower should be avoided.
        # velocities[velocities<10] = 0  

        return velocities
    
    def gradient(self, X, terrain, h=[0.001, 0.5], unit=False):
        """ Velocity gradient vector at given (cfa, pitch) and terrain using
            the Euclidean numerical derivative method
        
        Args:
            X (array_like): (2,) array of [cfa,pitch]
                (cfa in percentage, pitch in degrees)
            terrain (int): the MTTT terrain class (1...5). Terrain 1 is the fastest
                and 5 is the slowest (0 velocity). Default is 1.
            h (array_like): (2,) array of offsets for the numerical derivative
                along the cfa and pitch dimensions
            unit (bool): whether to return a gradient vector with unit norm

        Return:
            np.ndarray: gradient vector of size (2,)
        """

        X = np.array(X)
        if not self.in_domain(X):
            return

        grad = np.zeros(X.shape)
        for i in range(X.shape[0]):
            H = np.zeros(X.shape)
            H[i] = h[i]

            X_lower = X - H
            X_upper = X + H

            if (
                (not self.in_domain(X_lower, raise_error=False))
            ):
                # Use forward difference
                tmp = self.eval(X_upper,terrain)-self.eval(X, terrain)
                grad[i] = tmp/h[i]

            elif (
                (not self.in_domain(X_upper, raise_error=False))
            ):
                # Use backward difference
                tmp = self.eval(X,terrain)-self.eval(X_lower, terrain)
                grad[i] = tmp/h[i]

            else:
                # Use 2-point method
                tmp = self.eval(X_upper,terrain)-self.eval(X_lower, terrain)
                grad[i] = tmp/(2*h[i])
            
            log.debug(h, H, grad)
        
        # Turn into a unit vector if requested
        if unit:
            grad /= np.linalg.norm(grad)
        
        return grad
    
    def plot(self):
        """Plot the velocity function"""

        # Range of CFA and pitch values we're interested in
        cfa_vals = np.linspace(0,0.15,30)
        pitch_vals = np.linspace(-25,25,100)

        # Velocity for all [cfa, pitch] combinations
        pitch_grid, cfa_grid = np.meshgrid(pitch_vals,cfa_vals)
        all_cfa_pitch_inputs = np.hstack(
            (cfa_grid.flatten().reshape((-1,1)), 
            pitch_grid.flatten().reshape((-1,1)))
        )
        
        # Plot results
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15,3))

        velocity_grids = []
        vmin = np.inf
        vmax = 0
        for i, terrain in enumerate(self.terrain_domain):

            velocities = self.eval(all_cfa_pitch_inputs, terrain)
            velocities[velocities == 0] = np.nan
            velocity_grids.append(velocities.reshape(cfa_grid.shape).T)

            if not np.all(np.isnan(velocities)):
                vmin = min(np.nanmin(velocities), vmin)
                vmax = max(np.nanmax(velocities), vmax)

        for i, terrain in enumerate(self.terrain_domain):

            p = axes[i].pcolormesh(
                cfa_vals, pitch_vals, velocity_grids[i], shading='auto', 
                cmap='jet_r', vmin=10, vmax=vmax)
            fig.colorbar(p, ax=axes[i])

            # axes[i].set_aspect('equal', 'box')
            axes[i].set_ylabel("Rover pitch (deg)")
            axes[i].set_xlabel("CFA (fraction)")
            axes[i].set_title(f"Rover velocity (m/hr) \n(MTTT terrain {terrain})")

        plt.tight_layout()



class InverseGaussianPDF:

    velf = GaussianPDFVelocity()

    def __init__(self):
        pass
    
    def eval(self, features_dict):
        """ Cost rates given rover and terrain features
        
        Args:
            features_dict: dictionary of terrain and rover features
                (rover_pitch in rad, cfa within [0,1], terrain (default 1)).

        Return:
            np.ndarray: (n,) array of cost rates (hr/m)
        """

        velocities = self.velf.eval(features_dict)
        velocities[velocities == 0] = np.nan
        costs = 1/velocities

        return costs
    
    def gradient(self, X, terrain, h=[0.005, 0.5], unit=False):
        """ Velocity gradient vector at given (cfa, pitch) and terrain using
            the Euclidean numerical derivative method
        
        Args:
            X (array_like): (2,) array of [cfa,pitch]
                (cfa in percentage, pitch in degrees)
            terrain (int): the MTTT terrain class (1...5). Terrain 1 is the fastest
                and 5 is the slowest (0 velocity). Default is 1.
            h (array_like): (2,) array of offsets for the numerical derivative
                along the cfa and pitch dimensions
            unit (bool): whether to return a gradient vector with unit norm

        Return:
            np.ndarray: gradient vector of size (2,)
        """

        X = np.array(X)
        if not self.velf.in_domain(X):
            return

        grad = np.zeros(X.shape)
        for i in range(X.shape[0]):
            H = np.zeros(X.shape)
            H[i] = h[i]

            X_lower = X - H
            X_upper = X + H

            if (
                not self.velf.in_domain(X_lower, raise_error=False)
                or np.isnan(self.eval(X_lower, terrain))
            ):
                # Use forward difference
                tmp = self.eval(X_upper,terrain)-self.eval(X, terrain)
                grad[i] = tmp/h[i]

            elif (
                not self.velf.in_domain(X_upper, raise_error=False)
                or np.isnan(self.eval(X_upper, terrain))
            ):
                # Use backward difference
                tmp = self.eval(X,terrain)-self.eval(X_lower, terrain)
                grad[i] = tmp/h[i]

            else:
                # Use 2-point method
                tmp = self.eval(X_upper,terrain)-self.eval(X_lower, terrain)
                grad[i] = tmp/(2*h[i])
        
        # Turn into a unit vector if requested
        if unit:
            grad /= np.linalg.norm(grad)
        
        return grad
    
    def plot(self):
        """Plot the velocity function"""

        # Range of CFA and pitch values we're interested in
        cfa_vals = np.linspace(0,0.15,30)
        pitch_vals = np.linspace(-25,25,100)

        # Velocity for all [cfa, pitch] combinations
        pitch_grid, cfa_grid = np.meshgrid(pitch_vals,cfa_vals)
        all_cfa_pitch_inputs = np.hstack(
            (cfa_grid.flatten().reshape((-1,1)), 
            pitch_grid.flatten().reshape((-1,1)))
        )
        
        # Plot results
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15,3))

        cost_grids = []
        vmin = np.inf
        vmax = 0
        for i, terrain in enumerate(self.velf.terrain_domain):
            costs = self.eval(all_cfa_pitch_inputs, terrain)
            cost_grids.append(costs.reshape(cfa_grid.shape).T)

            if not np.all(np.isnan(costs)):
                vmin = min(np.nanmin(costs), vmin)
                vmax = max(np.nanmax(costs), vmax)

        for i, terrain in enumerate(self.velf.terrain_domain):
            p = axes[i].pcolormesh(
                cfa_vals,pitch_vals, cost_grids[i],
                shading='auto', cmap='jet', vmin=0.015, vmax=0.1)
            fig.colorbar(p, ax=axes[i])

            # axes[i].set_aspect('equal', 'box')
            axes[i].set_ylabel("Rover pitch (deg)")
            axes[i].set_xlabel("CFA (fraction)")
            axes[i].set_title(f"Rover cost (hr/m) \n(MTTT terrain {terrain})")

        plt.tight_layout()






    
