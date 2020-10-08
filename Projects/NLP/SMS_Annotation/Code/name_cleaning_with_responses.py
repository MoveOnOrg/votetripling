#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:49:04 2020

@author: alutes
"""

import argparse
from pathlib import Path
import pandas as pd
from utilities import clean_labeled_names

def main(args):
    
    # Set home directory
    home = Path(args.home_folder)
    DATA_FILE = Path(home, "Input_Data", args.data_file)

    # Read in
    data = pd.read_csv(DATA_FILE)

    # Fix NA Values
    data.loc[data.voterResponse.isnull(), 'voterResponse'] = ""
    data.loc[data.voterFinal.isnull(), 'voterFinal'] = ""
    data.loc[data.voterPost.isnull(), 'voterPost'] = ""
    data.loc[data.names.isnull(), 'names'] = ""

    # Only Retain relevant data
    data = data.loc[~(data.names == '')][['names', 'voterResponse', 'voterFinal', 'voterPost']]
    
    # Clean Names
    data['clean_names'] = ''
    for i, row in data.iterrows():
        names = row['names']
        response = row['voterResponse'] + ' ' + row['voterFinal'] + ' ' + row['voterPost']
        triple_message = row['tripleMessage']
        data.loc[i, 'clean_names'] = clean_labeled_names(names, response, triple_message)

    # Write out annotated file
    data.to_csv(Path(home, "Output_Data", args.output_file), index = False)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=(" ".join(__doc__.split("\n")[2:6])))
    PARSER.add_argument(
        "-f", "--home_folder", help="Location of home directory", type=str, required=False, default="./"
    )
    PARSER.add_argument(
        "-d", "--data_file", help="Name of of aggregated message file", type=str, required=False, default="labeled_agg.csv"
    )
    PARSER.add_argument(
        "-o", "--output_file", help="File name to dump output", type=str, required=False, default="labeled_names_cleaned_no_response.csv"
    )
    main(PARSER.parse_args())