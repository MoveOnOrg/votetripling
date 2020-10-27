#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:49:04 2020

@author: alutes
"""

import argparse
from pathlib import Path
from utilities import clean_labeled_names, load_civis, load_flat_file, export_civis

def main(args):
    
    # Set home directory
    home = Path(args.home_folder)

    # Read in data either from flat file or civis
    if args.use_civis:
        home = Path("./Projects/NLP/SMS_Annotation/")
        data = load_civis(args.input_data_filename.replace(".csv", ""), args.database_name)
    else:
        data = load_flat_file(home, args.input_data_filename)

    # Fix NA Values
    data.loc[data.triplemessage.isnull(), 'triplemessage'] = ""
    data.loc[data.voterresponse.isnull(), 'voterresponse'] = ""
    data.loc[data.voterfinal.isnull(), 'voterfinal'] = ""
    data.loc[data.voterpost.isnull(), 'voterpost'] = ""
    data.loc[data.names.isnull(), 'names'] = ""

    # Only Retain relevant data
    data = data.loc[~(data.names == '')]

    # Clean Names
    data['clean_names'] = ''
    data['review'] = False
    for i, row in data.iterrows():
        names = row['names']
        response = row['voterresponse'] + ' ' + row['voterfinal'] + ' ' + row['voterpost']
        clean_names, review = clean_labeled_names(names, response)
        data.loc[i, 'clean_names'] = clean_names
        data.loc[i, 'review'] = review

    # Write out annotated file
    if args.use_civis:
        export_civis(data, args.output_file.replace(".csv", ""), args.database_name)
    else:
        data.to_csv(Path(home, "Output_Data", args.output_file), index = False)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=(" ".join(__doc__.split("\n")[2:6])))
    PARSER.add_argument(
        "-f", "--home_folder", help="Location of home directory", type=str, required=False, default="./"
    )
    PARSER.add_argument(
        "-d", "--database_name", help="Name of database", type=str, required=False, default="Vote Tripling"
    )
    PARSER.add_argument(
        "-i", "--input_data_filename", help="Name of aggregated message file", type=str, required=False, default="labeled_agg.csv"
    )
    PARSER.add_argument(
        "-o", "--output_file", help="File name to dump output", type=str, required=False, default="labeled_names_cleaned_with_response.csv"
    )
    PARSER.add_argument(
        '-c', action='store_true', default=False, dest='use_civis', help='Whether to use civis for i/o'
    )
    main(PARSER.parse_args())