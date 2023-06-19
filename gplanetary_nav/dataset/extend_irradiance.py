#!/usr/bin/env python

""" 
    Artificially extend the irradiance coverage period by repeating
    the existing irradiance data & appending/chaining it to the current dataset
    (with corrected timestamps)

    usage: python extend_irradiance.py -h
    
    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

import shutil
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
from pathlib import Path

def extend_irradiance(
    dirpath: Path, repeat: int=1, file_prefix: str=None) -> None:
    """ Artificially extend the irradiance data by repeating the existing
        irradiance maps a given number of times & chaining them together

    Args:
        dirpath: absolute path to a dataset's root directory (which contains
            an 'irradiance' subdirectory)
        repeat: number of times to repeat the current dataset irradiance maps
            Default is 1.
        file_prefix: file name prefix for all the new irradiance maps created
    """

    dirpath = dirpath / "irradiance"
    file_prefix = '' if file_prefix is None else file_prefix
    
    times_fpath = dirpath / "times.csv"
    print(f"Using times.csv at: {times_fpath}")
    times_df = pd.read_csv(times_fpath)

    single_step = times_df['unix_timestamp_s'].diff().mean()
    cycle_step = times_df['unix_timestamp_s'].max() - times_df['unix_timestamp_s'].min()

    new_times = {k: list() for k in times_df.columns}
    for r in range(1, repeat+1):
        for _, row in times_df.iterrows():

            new_timestamp = int(row['unix_timestamp_s']+r*cycle_step+r*single_step)
            dt = datetime.utcfromtimestamp(new_timestamp)

            new_fname = dt.strftime(f'{file_prefix}_%Y-%jT%H-%M-%S.tif')
            new_date = dt.strftime('%Y-%m-%d')
            new_time = dt.strftime('%H-%M-%S')

            new_times["fname"].append(new_fname)
            new_times["UTC_date_yyyy-mm-dd"].append(new_date)
            new_times["UTC_time_HH-MM-SS"].append(new_time)
            new_times["unix_timestamp_s"].append(new_timestamp)

            shutil.copy(
                Path(dirpath, 'maps', row['fname']),
                Path(dirpath, 'maps', new_fname),
            )

        print(f"Repeated {r} time(s)")

    # Overwrite old times.csv
    new_times_df = pd.DataFrame(new_times)
    merged_times_df = pd.concat([times_df, new_times_df], ignore_index=True, axis=0)
    merged_times_df.to_csv(times_fpath, index=False)

    cnt = len(list(Path(dirpath, 'maps').glob('*')))
    print(f"Total number of maps: {cnt}. Check: {len(merged_times_df)==cnt}")
    print(f"New times.csv: \n {merged_times_df}")

def parse_args():
    # Retrieve input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dirpath', type=Path, required=True,
                        help='Full path to the irradiance directory')
    parser.add_argument('-r','--repeat', type=int, default=1,
                        help="Number of times to repeat existing maps")
    parser.add_argument('-f','--file_prefix', type=str, default=None,
                        help="File name prefix for the new irradiance maps")
    return parser.parse_args()

def main():

    parsed_args = parse_args()
    extend_irradiance(
        dirpath = parsed_args.dirpath,
        repeat = parsed_args.repeat,
        file_prefix= parsed_args.file_prefix)


if __name__=='__main__':
    main()
