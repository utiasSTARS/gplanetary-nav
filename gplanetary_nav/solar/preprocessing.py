#!/usr/bin/env python

""" 
    Solar irradiance preprocessing utilities

    Usage help: python preprocessing.py -h

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import shutil
import time
import csv
import json
from pathlib import Path
import sys
import argparse
import rasterio
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from gplanetary_nav.site import loader as site_loader
from gplanetary_nav.site.layers.nogo import NogoLayer

from gplanetary_nav.utils import split_array
from gplanetary_nav.solar.utils import fname_from_coord


def write_first_pixels(args):

    coord_list = args[0]
    save_dir = args[1]
    timestamp = args[2]
    irr_map = args[3]

    print("Started process...")
    sys.stdout.flush()

    for i_c, coords in enumerate(coord_list):

        # Create csv file
        with open(Path(save_dir, fname_from_coord(coords)), mode='w') as f:
            writer = csv.DictWriter(f, fieldnames=['unix_timestamp_s','flux_W_per_m2'])
            writer.writeheader()

            writer.writerow({
                'unix_timestamp_s' : timestamp,
                'flux_W_per_m2' : irr_map[coords]
            })
    
    return None

def create_pixel_intervals(
    dirpath: Path, num_cpus: int=1, create_nogo: bool=True,
    use_site_nogo: bool=True) -> None:
    """Create pixel-wise irradiance timeseries for a dataset containing
        irradiance maps
    
    Args:
        dirpath: absolute path to a dataset directory
        num_cpus: num of cpus to use for the timeseries initialization.
            Default 1
        create_nogo: create an nogo_irradiance.tif raster in the dataset's
            root directory to identify pixels where irradiance data isn't
            available (any pixel containing either a negative irradiance or
            a 'nodata' value as described in the geotiff metadata). This is
            carried out using only the first irradiance map, so it is assumed
            that all valid pixels on the first map also contain valid values
            on all other maps.
        use_site_nogo: create timeseries for pixels outside of the site layers
            no-go areas (in its current configuration as described in the
            dataset's settings.yaml file). Useful if the irradiance maps are
            large, but it means that a new set of irradiance timeseries needs
            to be calculated whenever the site layers are adjusted/changed.
    """

    # Load solar power mappings
    # irrmap = IrradianceMappingPercentage()

    # Load times of each layer
    times = dict()
    fpath = dirpath / "irradiance/times.csv"
    with open(fpath, mode='r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            times[row["fname"]] = row["unix_timestamp_s"]

    map_dir = Path(dirpath, "irradiance/maps")
    fpaths = sorted(map_dir.glob("*.tif"))

    # Create initial state of each pixel
    save_dir = Path(dirpath, "irradiance/pixel_intervals")

    # PART 1: Create all pixel interval files with starting values

    if save_dir.exists():
        print(f"Removing old directory {save_dir} (might take a few mins)")
        sys.stdout.flush()
        shutil.rmtree(save_dir)
    save_dir.mkdir()
    print(f"New directory created: {save_dir}")
    sys.stdout.flush()

    in_map_src = rasterio.open(fpaths[0])
    in_map = in_map_src.read(1)

    # Create irradiance no-go map
    irr_nogo = np.zeros(in_map.shape, dtype=np.uint8)
    irr_nogo[(in_map < 0) | (in_map == in_map_src.meta['nodata'])] = 1

    if create_nogo:
        out_meta = in_map_src.meta.copy()
        out_meta['nodata'] = None
        out_meta['dtype'] = NogoLayer.recommended_gtiff_dtype
        irr_nogo_layer = NogoLayer.from_raster(irr_nogo, out_meta)
        out_fpath = Path(dirpath, "nogo_irradiance.tif")
        irr_nogo_layer.save(out_fpath)
        print(f"Created & saved irradiance no-go file at {out_fpath}")
    
    if use_site_nogo:
        site = site_loader.load(dirpath)
        print(f"Loaded site at {dirpath}")
        sys.stdout.flush()
        irr_nogo[site.nogo.astype(np.bool)] = 1
        del site

    # Clip invalid pixels to 0 (to avoid overflow warnings, which are caused
    # by the nodata regions which we don't use anyways)
    in_map[irr_nogo.astype(np.bool)] = 0

    allowed_map = np.invert(irr_nogo.astype(np.bool))
    all_coords = list(map(tuple, np.argwhere(allowed_map).tolist()))
    print(f"Found {len(all_coords)} valid pixels")

    start_time = time.perf_counter()
    num_cpus = num_cpus
    print(f"Initializing all files with {num_cpus} parallel processes")
    sys.stdout.flush()

    coords_per_process = split_array(all_coords, num_cpus)
    args = tuple((coord_list, save_dir, times[fpaths[0].name],
        in_map) for coord_list in coords_per_process)

    with ProcessPoolExecutor(max_workers=num_cpus) as pool:
        results = pool.map(write_first_pixels, args)
    
    print(f"Initialization done (duration: {int(time.perf_counter()-start_time)}s)")
    sys.stdout.flush()

    # PART 2: Append to existing files

    max_irradiance = 0  # Keep track of maximum irradiance across all maps

    latest_map_src = rasterio.open(fpaths[0])
    latest_map = latest_map_src.read(1)
    latest_map[irr_nogo==1] = 0

    max_irradiance = max(max_irradiance, np.nanmax(latest_map))

    for i, fpath in enumerate(fpaths[1:], start=1):
        print(f"Processing map {i+1} of {len(fpaths)}")
        
        curr_time = times[fpath.name]
        curr_map_src = rasterio.open(fpath)
        curr_irr_map = curr_map_src.read(1)
        
        # Invalid pixels assumed to be in the shade
        curr_irr_map[irr_nogo==1] = 0

        # Find the pixel coordinates that need to be updated compared to latest info
        diff = latest_map-curr_irr_map
        mask = np.logical_and(diff!=0, allowed_map)
        changed_coords = list(map(tuple,np.argwhere(mask).tolist()))

        if len(changed_coords) == 0:
            continue

        max_irradiance = max(max_irradiance, np.nanmax(curr_irr_map))

        # Append changed values to csv
        for coords in changed_coords:
            with open(Path(save_dir, fname_from_coord(coords)), 'a') as f:
                writer = csv.DictWriter(f,
                    fieldnames=['unix_timestamp_s','flux_W_per_m2'])
                writer.writerow({
                    'unix_timestamp_s' : curr_time,
                    'flux_W_per_m2' : curr_irr_map[coords]
                })
        
        # Update the latest state of each (changed) pixel
        latest_map[mask] = curr_irr_map[mask]
    
    # Save max irradiance
    with open(Path(save_dir, 'max_irradiance.txt'), mode='w') as f:
        f.write(str(max_irradiance))
    print(f"Saved max irradiance achievable: {max_irradiance:.2f} W/m^2")

    # Save list of all preprocessed locations
    with open(Path(save_dir, 'valid_coords.json'), mode='w') as f:
        json.dump(all_coords, f)
    print(f"Saved all coordinates list")

def parse_args():

    # Retrieve input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dirpath', type=Path, required=True,
                        help='Full path to the site directory')
    parser.add_argument('-n','--num_cpus', type=int, default=5,
                        help=('Parallelize initialization across this number of '
                              'virtual CPUs'))
    parser.add_argument('-c', '--create_nogo', action='store_true', default=True,
                        help='Create & save irradiance no-go map')
    parser.add_argument('-s', '--use_site_nogo', action='store_true', default=True,
                        help='Limit timeseries creations to the site no-go map')
    return parser.parse_args()

def main():
    args = parse_args()

    create_pixel_intervals(
        dirpath = args.dirpath,
        num_cpus = args.num_cpus,
        create_nogo = args.create_nogo,
        use_site_nogo = args.use_site_nogo,
    )

if __name__ == "__main__":
    main()