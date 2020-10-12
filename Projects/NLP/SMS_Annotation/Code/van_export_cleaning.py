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
from utilities import cleanString, extract_good_tokens, \
                    get_token_features, get_doc, normalize_token, \
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

    # Get Extracted Names
    names_extract = []
    manual_review = []
    for i, row in van.iterrows():
        response = row['notetext']
        if (cleanString(response) == ""):
            names_extract.append("")
            manual_review.append(False)
            continue
        X_tokens_row = pd.DataFrame(
            get_token_features(response, row['contactname'], english_dict, census_dict, census_last_dict, token_counter)
            ).values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)
        doc = get_doc(response)
        clean_tokens = [normalize_token(t.string) for t in doc] 
        clean_tokens = [t for t in clean_tokens if not t == ""]
        
        # Extract any plausible tokens
        names_extract.append(extract_good_tokens(
                clean_tokens = clean_tokens, 
                triple_message = row['contactname'],
                y_pred = y_pred, 
                response = response, 
                threshold = LOWER_BOUND
                ))
        
        # Send to Manual Review if there are any tokens in the unclear range
        manual_review.append(((y_pred[:,1] > LOWER_BOUND) & (y_pred[:,1] < UPPER_BOUND)).sum() > 0)
    van['names_extract'] = names_extract
    van['manual_review'] = manual_review

    # Get those with confirmed names
    triplers = van.loc[(van.manual_review == False) & ~(van.names_extract == "")][['vanid', 'names_extract']]
    review = van.loc[van.manual_review == True][['vanid', 'contactname', 'notetext', 'names_extract']]
    
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
        "-d", "--database_name", help="Name of database", type=str, required=False, default="Vote_Tripling"
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