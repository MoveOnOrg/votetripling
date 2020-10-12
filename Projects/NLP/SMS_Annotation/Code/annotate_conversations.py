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
from utilities import featurize_conversation, add_token_features, load_civis, load_flat_file, export_civis
    
def main(args):

    # Set home directory
    home = Path(args.home_folder)

    # Read in data either from flat file or civis
    if args.use_civis:
        home = Path("./Projects/NLP/SMS_Annotation/")
        data = load_civis(args.input_data_filename.replace(".csv", ""))
        for col in ['noresponse', 'negresponse', 'posresponse', 'affirmresponse', 'finalaffirmresponse']:
            data[col] = data[col].astype(bool)
    else:
        data = load_flat_file(home, args.input_data_filename)
    
    # Thresholds for manual review and labeling
    LOWER_BOUND = .4 
    UPPER_BOUND = .75
    MID_BOUND = .5


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

    print("Cleaning and Featurizing...")

    # Fix NA Values
    data.loc[data.triplemessage.isnull(), 'triplemessage'] = ""
    data.loc[data.voterresponse.isnull(), 'voterresponse'] = ""
    data.loc[data.voterfinal.isnull(), 'voterfinal'] = ""
    data.loc[data.voterpost.isnull(), 'voterpost'] = ""

    # Number of tokens in final response
    data['num_tokens'] = data.voterfinal.str.count(" ") + ~(data.voterfinal == "")

    # Build Token Features
    data = add_token_features(data, token_model, english_dict, census_dict, census_last_dict, token_counter, threshold = LOWER_BOUND)

    # Build Features
    X = featurize_conversation(data, response_vectorizer, final_vectorizer, post_vectorizer)

    print("Annotating with Predictions...")

    # Add Predictions
    data['tripler_probability'] = model_tripler.predict_proba(X)[:, 1]
    data['name_provided_probability'] = model_name.predict_proba(X)[:, 1]
    data['optout_probability'] = model_opt.predict_proba(X)[:, 1]
    data['wrongnumber_probability'] = model_wrongnumber.predict_proba(X)[:, 1]

    # Create Dataset for triplers
    triplers = data.loc[
            (data.tripler_probability > UPPER_BOUND) &
            ((data.name_provided_probability > UPPER_BOUND) | (data.name_provided_probability < LOWER_BOUND)) &
            ((data.optout_probability > UPPER_BOUND) | (data.optout_probability < LOWER_BOUND)) &
            ((data.name_prob1 > UPPER_BOUND) | (data.name_prob1 < LOWER_BOUND)) &
            ((data.name_prob2 > UPPER_BOUND) | (data.name_prob2 < LOWER_BOUND)) &
            ((data.name_prob3 > UPPER_BOUND) | (data.name_prob3 < LOWER_BOUND))
            ].copy()
    triplers['is_tripler'] = 'yes'
    triplers.loc[triplers.name_provided_probability < UPPER_BOUND, 'names_extract'] = ''
    triplers['opted_out'] = np.where(triplers.optout_probability < UPPER_BOUND, 'no', 'yes')
    triplers['wrong_number'] = np.where(triplers.wrongnumber_probability < UPPER_BOUND, 'no', 'yes')
    triplers = triplers[['conversationid', 'contact_phone', 
                         'is_tripler', 'opted_out', 'wrong_number', 'names_extract']]

    # Create Dataset for manual review
    review = data.loc[
            ((data.tripler_probability < UPPER_BOUND) & (data.tripler_probability > LOWER_BOUND)) |
            ((data.name_provided_probability < UPPER_BOUND) & (data.name_provided_probability > LOWER_BOUND)) |
            ((data.optout_probability < UPPER_BOUND) & (data.optout_probability > LOWER_BOUND)) |
            ((data.name_prob1 < UPPER_BOUND) & (data.name_prob1 > LOWER_BOUND)) |
            ((data.name_prob2 < UPPER_BOUND) & (data.name_prob2 > LOWER_BOUND)) |
            ((data.name_prob3 < UPPER_BOUND) & (data.name_prob3 > LOWER_BOUND))
            ].copy()
    review['is_tripler'] = np.where(review.tripler_probability < MID_BOUND, 'no', 'yes')
    review.loc[review.name_provided_probability < MID_BOUND, 'names_extract'] = ''
    review['opted_out'] = np.where(review.optout_probability < MID_BOUND, 'no', 'yes')
    review['wrong_number'] = np.where(review.wrongnumber_probability < MID_BOUND, 'no', 'yes')
    review = review[['conversationid', 'contact_phone', 
                     'voterresponse', 'voterfinal', 'voterpost',
                     'is_tripler', 'opted_out', 'wrong_number', 'names_extract']]
    
    # Write out annotated files
    if args.use_civis:
        export_civis(triplers, args.output_filename.replace(".csv", ""))
        export_civis(review, args.manual_review_filename.replace(".csv", ""))
    else:
        triplers.to_csv(Path(home, "Output_Data", args.output_filename), index = False, encoding = 'latin1')
        review.to_csv(Path(home, "Output_Data", args.manual_review_filename), index = False, encoding = 'latin1')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=(" ".join(__doc__.split("\n")[2:6])))
    PARSER.add_argument(
        "-f", "--home_folder", help="Location of home directory", type=str, required=False, default="./"
    )
    PARSER.add_argument(
        "-d", "--input_data_filename", help="Name of of aggregated message file", type=str, required=False, default="testdata_aggregated.csv"
    )
    PARSER.add_argument(
        "-o", "--output_filename", help="File name to dump output", type=str, required=False, default='sms_triplers.csv'
    )
    PARSER.add_argument(
        "-m", "--manual_review_filename", help="File name to dump output", type=str, required=False, default='sms_manual_review.csv'
    )
    PARSER.add_argument(
        '-c', action='store_true', default=False, dest='use_civis', help='Whether to use civis for i/o'
    )
    main(PARSER.parse_args())