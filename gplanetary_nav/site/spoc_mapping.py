#!/usr/bin/env python

""" 
    Mapping to/from the raw SPOC terrain classes (Rothrock et al., 2016)

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""


# Terrain class labels of SPOC, according to Table 1 in Rothrock et al. (2016)
TERRAIN_LEGEND_SPOC = { 
    0: 'Smooth Regolith',
    1: 'Smooth Outcrop',
    2: 'Fractured Outcrop',
    3: 'Sparse Ripples (Firm Substrate)',
    4: 'Moderate Ripples (Firm Substrate)',
    5: 'Rough Regolith',
    6: 'Rough Outcrop',
    7: 'Sparse Ripples (Sandy Substrate)',
    8: 'Moderate Ripples (Sandy SUbstrate)',
    9: 'Dense Ridges',
    10: 'Rock Field',
    11: 'Solitary Ripple',
    12: 'Dense Linear Ripples',
    13: 'Sand Dune',
    14: 'Deep Sand (or Featureless sand)',
    15: 'Polygonal Ripples',
    16: 'Scarp or N/A',
    17: 'N/A',  # Some SPOC maps have a type 17, which I don't know
    255: 'N/A'
}

# Mapping from a SPOC terrain label to one of 5 MTTT traversability classes
# See Ono et al. (2018) for more information. Note that here, indexing of the
# terrain classes start at 0 due to package indexing convention
# Class 0 (1 in paper): Benign Terrains
# Class 1 (2 in paper): Rough Terrains
# Class 2 (3 in paper): Sandy Terrains
# Class 3 (4 in paper): No-autonav Terrains
# Class 4 (5 in paper): Untraversable

# MTTT terrain labels (see Ono et al. (2018))
# Note: labels start at 0 (instead of 1) due to terrain indexing conventions
TERRAIN_LEGEND_MTTT = {
    0 : 'Benign Terrains',
    1 : 'Rough Terrains',
    2 : 'Sandy Terrains',
    3 : 'No-autonav Terrains',
    4 : 'Untraversable',
    255: 'N/A'
}

SPOC_TO_MTTT_TRAVERSABILITY = {
    0: 0, 
    1: 0,
    2: 0,
    3: 2,
    4: 2,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 3,
    11: 4,
    12: 4,
    13: 4,
    14: 3,
    15: 4,
    16: 4,
    17: 255,  # Some SPOC maps have a type 17, which I don't know
    255: 255
}

MTTT_TRAVERSABILITY_TO_SPOC = {
    0: [0,1,2],
    1: [5,6],
    2: [3,4,7,8],
    3: [9,10,14],
    4: [11,12,13,15,16]
}

# Reduced terrain labels
# (lower resolution, but more convenient for engineering analysis)
TERRAIN_LEGEND_REDUCED = {
    0 : 'Cohesive Soil (CS)',
    1 : 'Bedrock (BR)',
    2 : 'Loose Soil (LS)',
    255: 'N/A'
}

# Mapping from the raw SPOC terrain labels to the reduced ones
SPOC_TO_REDUCED_TERRAIN = { 
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 0,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 255,
    17: 255,  # Some SPOC maps have a type 17, which I don't know
    255: 255
}
