#!/usr/bin/env python

""" 
    Custom types for global planning on a grid

    Authors: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from typing import Tuple
from dataclasses import dataclass


# A location on a grid (grid coordinates)
Node = Tuple[int,int]

@dataclass
class NodeStamped:
    # A location on a grid at a given time, typically a UNIX timestamp (s)
    node: Node
    time: float


