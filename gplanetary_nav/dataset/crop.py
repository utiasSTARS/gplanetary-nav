#!/usr/bin/env python

""" 
    Create cropped geotiffs or entire datasets from existing ones

    Usage: python crop.py -h

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from typing import List
from pathlib import Path
import argparse
import shutil
import rasterio
from rasterio.windows import Window
from pathlib import Path


def open_crop_save(
        in_fpath: Path, out_dirpath: Path, window: Window) -> None:
    """Open a source geotiff dataset, crop and save it. The output file name
    will be the same as the one in in_fpath.
    
    Args:
        in_fpath: absolute path to the input raster
        out_dirpath: absolute parent directory path for the cropped
            raster. Must already exist.
        window: cropping rasterio window
    """
    
    in_src = rasterio.open(in_fpath)
    
    out_tfm = in_src.window_transform(window)
    out_arr = in_src.read(window=window)
    
    out_meta = in_src.meta.copy()
    out_meta.update({
        'height':   out_arr.shape[1],
        'width':    out_arr.shape[2],
        'transform':out_tfm,
    })
    print(f"Processing {in_fpath}, new file shape: {out_arr.shape}")
    with rasterio.open(Path(out_dirpath, in_fpath.name), 'w', **out_meta) as f:
        f.write(out_arr)
    
def new_dataset_from_window(
        in_dirpath: Path, out_dirpath: Path, window: Window,
        ignore: List[str]=['.xml']) -> None:
    """Create a new dataset from an existing one using a cropping window

    Args:
        in_dirpath: absolute directory path of the dataset to crop from
        out_dirpath: absolute parent directory path for the cropped raster.
        window: cropping rasterio window
        ignore: file extensions to ignore, including the leading period
            (ex: ['.xml','.pkl']). Default is ['.xml']
    """
    
    out_dirpath.mkdir()  # FileExistsError if out_dirpath already exists
    
    # Site layers
    files_in_root = [f for f in in_dirpath.glob('*') if f.is_file()]
    for in_fpath in files_in_root:
        if in_fpath.suffix == '.tif':
            open_crop_save(in_fpath, out_dirpath, window)
        elif in_fpath.suffix not in ignore:
            out_fpath = Path(out_dirpath, in_fpath.name)
            shutil.copyfile(in_fpath, out_fpath) 
        else:
           pass

    # Illumination maps, if any
    in_irr_dirpath = Path(in_dirpath, "irradiance")
    if in_irr_dirpath.exists():
        out_irr_dirpath = Path(out_dirpath, "irradiance")
        out_irr_dirpath.mkdir()
        
        out_map_dirpath = Path(out_irr_dirpath, "maps")
        out_map_dirpath.mkdir()

        shutil.copyfile(
                Path(in_irr_dirpath, "times.csv"),
                Path(out_irr_dirpath, "times.csv")
        )
        
        for in_fpath in Path(in_dirpath, "irradiance", "maps").glob('*.tif'):
            open_crop_save(in_fpath, out_map_dirpath, window)

def parse_args():

    # Retrieve input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--in_dirpath', type=Path, required=True,
                        help='Complete input dataset directory path')
    parser.add_argument('-o','--out_dirpath', type=Path, required=True,
                        help='Complete output dataset directory path')
    parser.add_argument('-r','--row_off', type=int, required=True,
                        help='Row index of the top left window corner')
    parser.add_argument('-c','--col_off', type=int, required=True,
                        help='Col index of the top left window corner')
    parser.add_argument('-w','--width', type=int, required=True,
                        help='Width of the window, in pixels')
    parser.add_argument('-e','--height', type=int, required=True,
                        help='Height of the window, in pixels')
    parser.add_argument('-g','--ignore', nargs='+', default=['.xml'],
                        help=(  'File extensions to ignore with leading period'
                                '. Ex: ... -g .xml .pkl'))
    return parser.parse_args()


def main():
    
    args = parse_args()
    
    window = Window(
            col_off=args.col_off,
            row_off=args.row_off,
            width=args.width,
            height=args.height
    )
    
    new_dataset_from_window(
            in_dirpath=args.in_dirpath,
            out_dirpath=args.out_dirpath,
            window=window,
            ignore=args.ignore
    )

    
if __name__ == "__main__":
    main()