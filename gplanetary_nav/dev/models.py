#!/usr/bin/env python

""" 
    Dummy models for the development of the gplanetary_nav package

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

DUMMY_ROVER_CFG = {
    'solar_panel': {
        'area': 1.5, # m^2
        'efficiency': 0.3, # between 0 and 1 (0%-100%)
    },
    'motion': {
        'velocity': 0.045, # m/s
        'power': 137, # W
    },
}