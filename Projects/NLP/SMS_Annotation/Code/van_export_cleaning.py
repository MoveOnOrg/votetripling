#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:33:21 2020

@author: alutes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:44:24 2020

@author: alutes
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from utilities import featurize_conversation_van, add_token_features_van, \
                    load_civis, load_flat_file, export_civis
    
def main(args):

    # Set home directory
    home = Path(args.home_folder)
    
    # Read in data either from flat file or civis
    if args.use_civis:
        home = Path("./Projects/NLP/SMS_Annotation/")
        van = load_civis(args.input_data_filename.replace(".csv", ""), args.database_name)
    else:
        van = load_flat_file(home, args.input_data_filename)
    
    # Thresholds for manual review and labeling
    LOWER_BOUND = .4
    UPPER_BOUND = .75

    print("Loading Models...")


    pickle_file = Path(home, "Models", "annotation_models.pkl")
    with open(pickle_file, "rb") as f:
        # N-Gram Featurizers
        response_vectorizer = pickle.load(f)
        final_vectorizer = pickle.load(f)
        post_vectorizer = pickle.load(f)

        # Logistic Regressions
        token_model = pickle.load(f)
        model_tripler = pickle.load(f)
        model_name = pickle.load(f)
        model_opt = pickle.load(f)
        model_wrongnumber = pickle.load(f)
        token_counter = pickle.load(f)
        model_van_name = pickle.load(f)
        van_vectorizer = pickle.load(f)
        Features = pickle.load(f)
        model_token_bow = pickle.load(f)
        van_token_vectorizer = pickle.load(f)

    print("Loading Data...")

    # US Census Data
    census = pd.read_csv(Path(home, "Utility_Data", "census_first_names_all.csv"))
    census_dict = {}
    for i, row in census.iterrows():
        census_dict[row['name']] = np.log(row['census_count'])

    # Last Name Data
    census_last = pd.read_csv(Path(home, "Utility_Data", "census_last_names_all.csv"))
    census_last_dict = {}
    for i, row in census_last.iterrows():
        census_last_dict[row['name']] = np.log(row['census_count'])

    # US Word Freq Data
    english = pd.read_csv(Path(home, "Utility_Data", "english.csv"))
    english_dict = {}
    for i, row in english.iterrows():
        english_dict[row['name']] = row['freq']

    # Clean NA values
    van.loc[van.notetext.isnull(), 'notetext'] = ""
    van.loc[van.contactname.isnull(), 'contactname'] = ""

    # Number of tokens
    van['num_tokens'] = van.notetext.str.count(" ") + ~(van.notetext == "") 

    # Build Token Features
    van = add_token_features_van(van, 
                                 van_token_vectorizer, model_token_bow,
                                 token_model, Features,
                                 english_dict, census_dict, 
                                 census_last_dict, 
                                 token_counter,
                                 LOWER_BOUND = LOWER_BOUND,
                                 UPPER_BOUND = UPPER_BOUND)

    # Build Features
    X = featurize_conversation_van(van, van_vectorizer)

    print("Annotating with Predictions...")

    # Add Predictions
    van['names_probability'] = model_van_name.predict_proba(X)[:, 1]

    # Get those with confirmed names
    triplers = van.loc[(van.manual_review == False) & 
                       ~(van.names_extract == "") &
                       (van.names_probability > UPPER_BOUND)][['voter_file_vanid', 'names_extract']]
    review = van.loc[(van.names_probability > LOWER_BOUND) &
        (
         (van.manual_review == True) |
         (van.names_probability < UPPER_BOUND)
         )][['voter_file_vanid', 'contactname', 'notetext', 'names_extract']]
    
    # Write out annotated files
    if args.use_civis:
        export_civis(triplers, args.output_filename.replace(".csv", ""), args.database_name)
        export_civis(review, args.manual_review_filename.replace(".csv", ""), args.database_name)
    else:
        triplers.to_csv(Path(home, "Output_Data", args.output_filename), index = False, encoding = 'latin1')
        review.to_csv(Path(home, "Output_Data", args.manual_review_filename), index = False, encoding = 'latin1')

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=(" ".join(__doc__.split("\n")[2:6])))
    PARSER.add_argument(
        "-f", "--home_folder", help="Location of home directory", type=str, required=False, default="./"
    )
    PARSER.add_argument(
        "-d", "--database_name", help="Name of database", type=str, required=False, default="Vote Tripling"
    )
    PARSER.add_argument(
        "-i", "--input_data_filename", help="Name of aggregated message file", type=str, required=False, default="van_export.csv"
    )
    PARSER.add_argument(
        "-o", "--output_filename", help="File name to dump output", type=str, required=False, default='van_cleaned.csv'
    )
    PARSER.add_argument(
        "-m", "--manual_review_filename", help="File name to dump output", type=str, required=False, default='van_manual_review.csv'
    )
    PARSER.add_argument(
        '-c', action='store_true', default=False, dest='use_civis', help='Whether to use civis for i/o'
    )
    main(PARSER.parse_args())