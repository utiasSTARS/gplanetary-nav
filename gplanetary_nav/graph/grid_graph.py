#!/usr/bin/env python

""" Create a directed graph with features from a provided site.

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from __future__ import annotations
from typing import Union, Tuple
from pathlib import Path
import logging
import json
import hashlib

import numpy as np
import networkx as nx
from gplanetary_nav.utils import load_yaml_with_includes
from gplanetary_nav.site import loader as site_loader

# from gplanetary_nav.cost.time.constant import ConstantVelocity
# from gplanetary_nav.cost.energy.constant_power import ConstantPower

from gplanetary_nav.cost import time as time_cost
from gplanetary_nav.cost import energy as energy_cost

from gplanetary_nav.planning.types import Node


log = logging.getLogger(__name__)


# Grid action delta for a 4-connected grid
COORD_DELTA_4 = {
    'N' : (-1,0),
    'E' : (0,1),
    'S' : (1,0),
    'W' : (0,-1)
}

# Grid action delta for a 8-connected grid
COORD_DELTA_8 = {
    **COORD_DELTA_4,
    **{
        'NE' : (-1,1),
        'SE' : (1,1),
        'SW' : (1,-1),
        'NW' : (-1,-1)
    }
}

# Azimuth angle (radians) from North, increasing clockwise (towards East, etc.)
AZIMUTH_FROM_NORTH = {
    'N' : 0.0,
    'E' : np.pi/2,
    'S' : np.pi,
    'W' : 3*np.pi/2,
    'NE' : np.pi/4,
    'SE' : 3*np.pi/4,
    'SW' : 5*np.pi/4,
    'NW' : 7*np.pi/4
}

# 2D grid distance (unitless) from heading direction
GRID_DISTANCE = {
    'N' : 1.0,
    'E' : 1.0,
    'S' : 1.0,
    'W' : 1.0,
    'NE' : np.sqrt(2.0),
    'SE' : np.sqrt(2.0),
    'SW' : np.sqrt(2.0),
    'NW' : np.sqrt(2.0)
}


class GridGraph:
    def __init__(
        self, world_cfg: dict, rover_cfg: dict) -> None:
        """Generate a grid graph with terrain features given a site

        Args:
            world_cfg: the world configuration dictionary. Must contain
                a 'site_dirpath' attribute and a 'graph' subdictionary
            rover_cfg: the rover configuration dictionary
        """

        # Load site
        self.world_cfg = world_cfg

        # if 'site_dirpath' not in self.world_cfg.keys():
        #     self.world_cfg['site_dirpath'] = str(Path(
        #         os.environ.get('GNAV_DATASET_PATH'),
        #         self.world_cfg['site_name']))
        self.rover_cfg = rover_cfg

        # print(self.world_cfg['graph'])

        # self.site = site
        # self.neighborhood = neighborhood

        # # Check inputs
        # if neighborhood not in [4,8]:
        #     log.error(f"Invalid neighborhood ({neighborhood}), need 4 or 8")
        # else:
        #     log.info(f'Initiated weighted graph for site {site.name} '
        #              f'and neighborhood/connectivity of {neighborhood}')
    
    @classmethod
    def load(cls, world_cfg: dict, rover_cfg: dict) -> nx.DiGraph:
        """Load a graph

        Args:
            world_cfg: world configuration
            rover_cfg: rover configuration
        """

        inst = cls(world_cfg, rover_cfg)
        if not inst.load_data():
            log.warn(
                f"No matching graph found, generating a new one")
            inst.create_and_save(ret_G=False)
        
        return inst
    
    def load_data(self) -> bool:
        """Load a graph
        
        Return:
            True if a matching config was found, False otherwise
        """

        save_dir = Path(
            self.world_cfg['site_dirpath'], 'graph', self.get_checksum())
        if not save_dir.exists():
            return False

        self.G = nx.read_gpickle(Path(save_dir, 'G.pkl'))
        log.info(f"Loaded graph with {len(self.G.nodes)} nodes "
              f"& {len(self.G.edges)} edges")
        return True
        
    def create_and_save(self, ret_G: bool=False) -> None | nx.DiGraph:
        """Create a dense directed graph with edge features.

        Args:
            ret_G: whether to return graph. Default is True
        
        Returns:
            the graph if ret_G is True, otherwise None
        """

        self.site = site_loader.load(self.world_cfg['site_dirpath'])

        # Initiate directed weighted graph
        self.G = nx.DiGraph()

        if self.world_cfg['graph']['connectivity'] == 4:
            coord_delta = COORD_DELTA_4
        elif self.world_cfg['graph']['connectivity'] == 8:
            coord_delta = COORD_DELTA_8
        else:
            raise NotImplementedError(
                f"Invalid graph connectivity: ("
                f"{self.world_cfg['graph']['connectivity']}), need 4 or 8")

        # Go by driving heading
        for heading, delta in coord_delta.items():
            print(f"Processing edges with heading {heading}")

            for orig_node in map(tuple,
                    np.argwhere(np.invert(self.site.nogo.astype(np.bool)))):

                dest_node = tuple(orig_node[i] + delta[i] for i in range(2))
                if not self.site.in_bounds(dest_node):
                    continue

                # Ensure to_node is not out of range or not in an obstacle
                if not self.site.in_nogo(dest_node):

                    features = dict()

                    # Total edge length
                    l = self.edge_length(orig_node, dest_node, heading)
                    features['length'] = l

                    # Pitch & roll for the edge segment overlapping with
                    # the origin node
                    (p, r) = self.pitch_roll(orig_node, heading)
                    features['pitch_orig'] = p
                    features['roll_orig'] = r

                    # Pitch & roll for the edge segment overlapping with
                    # the destination node
                    (p, r) = self.pitch_roll(dest_node, heading)
                    features['pitch_dest'] = p
                    features['roll_dest'] = r

                    # Underlying terrain properties at the originating node
                    # for layer in self.site.layers:
                    #     # Skip no-go, mosaic or elevation data
                    #     if 'nogo' in layer:
                    #         continue
                    #     elif layer in ['mosaic','dem']:
                    #         continue

                    #     features[layer] = getattr(self.site, layer)[from_node]

                    self.G.add_edge(orig_node, dest_node, **features)
        
        # Weigh edges
        if 'velocity_model' in self.world_cfg['graph']:
            # Add 'time' edge attributes
            # if self.world_cfg['graph']['velocity_model']['name'] == 'ConstantVelocity':
            #     cost = time_cost.ConstantVelocity(
            #         self.rover_cfg['motion']['velocity'])
            # else:
            CostClass = getattr(time_cost, self.world_cfg['graph']['velocity_model']['name'])
            custom_params = self.world_cfg['graph']['velocity_model']['params']
            custom_params = dict() if custom_params is None else custom_params
            
            cost = CostClass(self.site, self.rover_cfg, **custom_params)
            self.weigh_edges('time', cost)
        
        if 'power_model' in self.world_cfg['graph']:
            # Add 'energy' edge attributes
            # if self.world_cfg['graph']['power_model']['name'] == 'ConstantPower':
            #     cost = energy_cost.ConstantPower(
            #         self.rover_cfg['motion']['power'])
            # else:
            CostClass = getattr(energy_cost, self.world_cfg['graph']['power_model']['name'])
            custom_params = self.world_cfg['graph']['power_model']['params']
            custom_params = dict() if custom_params is None else custom_params
            
            cost = CostClass(self.site, self.rover_cfg, **custom_params)
            self.weigh_edges('energy', cost)

        self.save_graph()
        
        if ret_G:
            return self.G

    def edge_length(
        self, f_node: Node, t_node: Node, heading: str) -> float:
        """Euclidean (3D) distance between two neighbouring node centers

        Args:
            f_node: 'from' (origin) node grid coordinate
            t_node: 'to' (destination) node grid coordinate
            heading: edge heading in cardinal code (ex: 'N', 'NE', 'E', etc.)
        
        Returns:
            edge length in meters
        """

        # Calculate 3D physical length of edge
        d_elev = np.abs(self.site.dem[f_node]-self.site.dem[t_node])
        d_xyplane = self.site.resolution_mtr*GRID_DISTANCE[heading]
        length = np.sqrt(d_elev**2+d_xyplane**2)
        return length
    
    def pitch_roll(self, node: Node, heading: str) -> Union[float, float]:
        """Rover pitch & roll at a pixel location with a given heading

        Args:
            node: tuple grid coordinates
            heading: edge heading in cardinal code (ex: 'N', 'NE', 'E', etc.)
        
        Return:
            rover pitch and roll, in radians
        """

        # Find attack angle and limit it between -pi <= attack < pi
        # Note: angle of attack of 0 is downhill, +ve attack is 'left' when
        # facing downhill, -ve attack is 'right' when facing downhill
        attack = self.site.aspect[node] - AZIMUTH_FROM_NORTH[heading]
        attack = np.fmod(attack + np.pi, 2*np.pi) - np.pi

        # Effective pitch of the rover (rotation about y-axis, pointing towards
        # the right of the rover) for the given angle of attack.
        # A pitch of 0 means the rover is driving cross-slope (angle of attack
        # is either pi/2 or -pi/2)
        # A +ve pitch means that the rover is driving uphill, while a negative
        # pitch means it's driving downhill
        pitch = self.site.slope[node]*np.sin(attack-np.pi/2)

        # Roll of rover (with respect to robot's x-axis, pointing forward)
        # A roll of 0 means the rover is driving strictly uphill or downhill.
        # A +ve roll means that the rover is 'leaning to its right' while a -ve
        # roll means that the rover is 'leaning to its left'.
        # Note: roll and pitch angle of rover are phased out by pi/2
        roll = pitch*np.sin(attack)

        return pitch, roll

    
    # def eval_path(self, path: List[Tuple[int,int]], weight: str) -> float:
    #     """Evaluate a path cost
        
    #     Args:
    #         path: list of (consecutive) nodes
    #         weight: edge feature to use as cost
        
    #     Return:
    #         total path cost
    #     """

    #     total_cost = 0
    #     for n0, n1 in pairwise(path):
    #         total_cost += self.G[n0][n1][weight]
        
    #     return total_cost
    
    # def get_bresenham_path(self, path, clean_path=True):
    #     """ Take a low-resolution path and return an approximately similar path
    #         along an 8-connected grid graph using the Bresenham line algorithm
        
    #     Args:
    #         path (list): list of nodes along input path
    #         clean_path (bool): remove 4-directions zig-zag cardinal motions 
    #             and enforce diagonal connectivity by performing one pass over
    #             the raw bresenham trajectory. Default is True.
        
    #     Returns:
    #         list: list of nodes of the approximately similar 8-connected path
    #     """

    #     if self.neighborhood != 8:
    #         raise Warning("The grid graph needs a 8-neighbours connectivity")
        
    #     # Raw Bresenham path
    #     bresenham_path = []
    #     for i, e in enumerate(pairwise(path)):
    #         line = bresenham_line(tuple(map(int, e[0])), tuple(map(int, e[1])))
    #         if i == len(path)-2:
    #             bresenham_path += line
    #         else:
    #             bresenham_path += line[:-1]
        
    #     # Make sure none of the paths go through a no-go region
    #     for i, node in enumerate(bresenham_path):
    #         if self.site.in_nogo(node):
    #             for delta in COORD_DELTA_8.values():

    #                 node_candidate = tuple(np.array(node)+np.array(delta))

    #                 # Ensure not in nogo
    #                 if self.site.in_nogo(node_candidate):
    #                     continue
                    
    #                 # Ensure it can connect to the previous node and next node
    #                 prev_node = bresenham_path[i-1]
    #                 d = np.abs(np.array(node_candidate)-np.array(prev_node))
    #                 if not np.all(d<=1):
    #                     continue
                    
    #                 # Ensure it can connect to the previous node and next node
    #                 next_node = bresenham_path[i+1]
    #                 d = np.abs(np.array(node_candidate)-np.array(next_node))
    #                 if not np.all(d<=1):
    #                     continue
                    
    #                 # Found new node if it made it this far
    #                 break
            
    #             bresenham_path.insert(i, node_candidate)
    #             bresenham_path.remove(node)

    #     if not clean_path:
    #         return bresenham_path
        
    #     # Cleanup operation
    #     nodes_idxs_adjacent_to_removed = []
    #     path_cleaned = False
    #     while not path_cleaned:
    #         path_cleaned = True
    #         for i, e_pair in enumerate(pairwise(pairwise(bresenham_path))):
    #             node0 = e_pair[0][0]
    #             node1 = e_pair[0][1]
    #             node2 = e_pair[1][1]

    #             node0_idx = i
    #             node1_idx = i+1
    #             node2_idx = i+2

    #             # Do not remove two adjacent nodes
    #             if node1_idx in nodes_idxs_adjacent_to_removed:
    #                 continue

    #             if np.all(np.abs(np.array(node0)-np.array(node2)) <= 1):
    #                 del bresenham_path[node1_idx]

    #                 # After node1 deletion, the index of node2 decreases by 1
    #                 nodes_idxs_adjacent_to_removed += [node0_idx, node2_idx-1]
    #                 path_cleaned = False
    #                 break
        
    #     # Remove duplicated nodes caused by the cleanup operation
    #     duplicated = []
    #     for a in pairwise(bresenham_path):
    #         if a[0] == a[1]:
    #             duplicated.append(a[0])
    #     for dupl in duplicated:
    #         bresenham_path.remove(dupl)
        
    #     return bresenham_path
    
    def weigh_edges(self, prefix: str, cost: object):
        """Weigh edges based on a cost function using the edge origin and
            destination properties

        Args:
            prefix: prefix of the new edge cost features created. <prefix>_orig
                and <prefix>_dest cost features will be created with the edge 
                origin and destination properties, respectively. <prefix>_total
                will store the sum of both (i.e. the 'total' cost).
            cost: the cost model. Must contain an 'eval()' that returns
                the cost corresponding to a given set of edge properties.
                This method must accept the following arguments:
                    node: pixel grid coordinates where drive occurs
                    pitch: rover pitch in radians (see sign convention above)
                    roll: rover roll in radians (see sign convention above)
                    length: length of the edge segment
                The model/object must also contain a 'meta' attribute/property
                returning a string
        """

        for orig_node, dest_node, feat_dict in self.G.edges(data=True):
            
            try:
                time = feat_dict['time_orig']
            except KeyError:
                time = None
            
            feat_dict[f'{prefix}_orig'] = cost.eval(
                node=orig_node,
                pitch=feat_dict['pitch_orig'],
                roll=feat_dict['roll_orig'],
                length=feat_dict['length']/2,
                duration=time)

            try:
                time = feat_dict['time_dest']
            except KeyError:
                time = None
            
            feat_dict[f'{prefix}_dest'] = cost.eval(
                node=dest_node,
                pitch=feat_dict['pitch_dest'],
                roll=feat_dict['roll_dest'],
                length=feat_dict['length']/2,
                duration=time)
            
            feat_dict[f'{prefix}_total'] = feat_dict[f'{prefix}_orig'] + \
                feat_dict[f'{prefix}_dest']

            self.G.add_edge(orig_node, dest_node, **feat_dict)


    def save_graph(self):
        """Save graph in root dataset directory as G<neighborhood>.pkl and
            corresponding metadata in G<neighborhood>_metadata.txt
        """

        graph_dir = Path(self.site.dirpath, 'graph')
        graph_dir.mkdir(exist_ok=True)

        save_dir = Path(graph_dir, self.get_checksum())
        save_dir.mkdir()

        # Save graph
        fpath = Path(save_dir, "G.pkl")
        log.info(f"Saving graph to {fpath}")
        nx.write_gpickle(self.G, fpath)

        # Save metadata
        fpath = Path(save_dir, "G_metadata.txt")
        log.info(f"Saving graph metadata to {fpath}")
        with open(fpath, 'w') as f:
            f.write(f"Graph grid connectivity: {self.world_cfg['graph']['connectivity']}")
            f.write(f'\nNum of nodes: {len(self.G.nodes)}')
            f.write(f'\nNum of edges: {len(self.G.edges)}')

            for _, _, features_dict in self.G.edges(data=True):
                break
            f.write(f'\nEdge attributes: {list(features_dict.keys())}')
            f.write('\n')
    
    def get_checksum(self):
        """MD5 checksum of current configuration"""
        data = dict(self.rover_cfg['motion'])
        data['graph'] = self.world_cfg['graph']

        # Keep track of site settings (could cause changes in nogo map)
        fpath = Path(self.world_cfg['site_dirpath'], 'settings.yaml')
        data['site_settings'] = load_yaml_with_includes(fpath)

        md5_checksum = hashlib.md5(
            json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()
        return md5_checksum

