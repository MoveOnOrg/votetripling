#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:44:24 2020

@author: alutes
"""
import argparse
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from scipy.sparse import hstack
import spacy
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
nlp = spacy.load('en')
AFFIXES = "\\b(mr|mrs|ms|dr|jr|sr|your|her|his|our|their|in|you)\\b"
POSSESSIVES = "\\b(my|his|her|their|our|step)\\b"
RELATIONSHIPS = "\\b((step|grand)[- ]?)?(house|kid|aunt|uncle|partner|ma|pa|boss[a-z]+|follower|sibling|brother|sister|son|daughter|children|child|kid|parent|mom|mother|dad|father|friend|family|cowor[a-z]+|colleague|church|pastor|priest|[a-z]*mate|husband|wife|spouse|girlfriend|boyfriend|neighbo[a-z]+|in[ -]?law)[s]*\\b"
EXCLUDE = "\\b(kamala|biden|member|trump|donald|melania|ivanka|idk|ty|yw|yay|oops|ooops|yes[a-z]+|ah|ill|o|y|lol|jr|sr|sir|dr|mr|mrs|ms|dr|dude|ditto|tmi|jk|rofl)\\b"
EXCLUDE_PRIOR = "\\b(im|vote for|my name is|this is|who is|this isnt|not|support)\\b"
NEW_LINE_REG = "\\n|\n|\\\\n"

################################
# Basic Cleaning Functions
################################

def normalize_token(token):
    return re.sub("[^a-z]", "", token.lower().strip())

def get_doc(voterResponse):
    voterResponseClean = re.sub("\\n", " ", voterResponse)
    voterResponseClean = re.sub("\\.", ". ", voterResponseClean)
    voterResponseClean = re.sub(",", ", ", voterResponseClean)
    voterResponseClean = re.sub("\\&", " and ", voterResponseClean)
    voterResponseClean = re.sub("\\s+", " ", voterResponseClean)
    voterResponseCamelized = re.sub("([a-z][a-z]+)([A-Z])", "\\1 \\2", voterResponseClean)
    return nlp(voterResponseCamelized)

def cleanString(string, exclude_reg = '\\&|\\band\\b|\\bmy\\b'):
    noAnd = re.sub(exclude_reg, ' ', string.lower())
    noChar = re.sub('[^a-z ]', ' ', noAnd)
    return re.sub('\\s+', ' ', noChar).strip()

def get_list(lst, index):
    if index < 0 or index >= len(lst):
        return ''
    else:
        return lst[index]

def present_tokens(clean_tokens, 
                response,
                triple_message,
                excluded = EXCLUDE,
                possessive = POSSESSIVES,
                relations = RELATIONSHIPS):
    triple_tokens = cleanString(triple_message).split(" ")
    good_tokens = []
    for j, token in enumerate(clean_tokens):
        # For relationships, look for the proper modifier
        if re.match(relations, token):
            pos_match = re.search("\\b(his|her|their|step) %s"%token, response)
            if pos_match:
                token = pos_match.group()
            else:
                token = "your " + token
        elif re.match(possessive, token) is not None or \
             re.match(excluded, token) is not None or \
             token in triple_tokens:
             continue
        # For names, capitalize
        else:
            token = token.capitalize()
        good_tokens.append(token)
        
    name_tokens = list(set(good_tokens))
    if len(name_tokens) < 1:
        return ''
    name_tokens[len(name_tokens) - 1] = 'and ' + name_tokens[len(name_tokens) - 1]
    return ', '.join(name_tokens)
    
def extract_good_tokens(clean_tokens, 
                        triple_message,
                        y_pred, 
                        response,
                        threshold = .5):
    good_tokens = [t for j, t in enumerate(clean_tokens) if y_pred[j, 1] > threshold]
    return present_tokens(good_tokens, response, triple_message)

################################
# Featurizing Functions
################################
        
# Get features for a single token
def get_token_features(voterResponse, 
                       tripleMessage,
                       english_dict,
                       census_dict,
                       census_last_dict,
                       token_counter,
                       smooth_census = 0,
                       smooth_eng = 1,
                       relationship_reg = RELATIONSHIPS,
                       and_reg = "\\band\\b|\\by\\b|\\&|\\by\\b",
                       sep_reg = "\\||,|-|\\.",
                       possessive_reg = POSSESSIVES,
                       excludelist_reg = EXCLUDE,
                       exclude_prior_reg = EXCLUDE_PRIOR,
                       is_initial_response = True,
                       name_threshold = 8
                       ):
    token_features = []

    # Parse using spacy
    doc = get_doc(voterResponse)
    persons = [e for e in doc.ents if e.label_ == "PERSON"]
    clean_tokens = [normalize_token(t.string) for t in doc]
    cleaned_string = ' '.join(clean_tokens)
    is_wordlike = [not t == "" for t in clean_tokens]
    is_seperator = [re.match(sep_reg, t.string.lower()) is not None for t in doc]
    is_and = [re.match(and_reg, t.string.lower()) is not None for t in doc]
    is_capital = [re.match('[A-Z]', t.string) is not None for t in doc]
    is_relationship = [re.match(relationship_reg, t) is not None for t in clean_tokens]
    is_exclude = [re.match(excludelist_reg, t) is not None for t in clean_tokens]
    is_possessive = [re.match(possessive_reg, t) is not None for t in clean_tokens]
    name_probs = [census_dict.get(t, smooth_census) for t in clean_tokens]   
    triple_message_tokens = cleanString(tripleMessage).split(" ")

    # Comment Level Features
    total_tokens = len(doc)
    word_tokens = sum(is_wordlike)
    and_tokens = sum(is_and)
    seperators = sum(is_seperator)
    other = total_tokens - word_tokens - seperators

    for j, clean_token in enumerate(clean_tokens):

        # Skip Non-Wordlike tokens
        if not is_wordlike[j]:
            continue

        # Build all token features
        feature_dict = {
                'initialResponse' : is_initial_response,
                'token_in_triple_message' : clean_token in triple_message_tokens,
                'token_length' : len(clean_token),
                'position' : j,
                'num_tokens' : total_tokens,
                'word_tokens' : word_tokens,
                'relation_tokens' : sum(is_relationship),
                'name_tokens' : np.sum(np.array(name_probs) > name_threshold),
                'and_tokens' : and_tokens,
                'seperators' : seperators,
                'other' : other,
                'relationship' : is_relationship[j],
                'exclude' : is_exclude[j],
                'exclude_phrase' : re.match(exclude_prior_reg + ' ' + clean_token, cleaned_string) is not None,
                'eng_prob' : english_dict.get(clean_token, smooth_eng),
                'name_prob' : name_probs[j],
                'corpus_prob' : np.log(token_counter.get(stemmer.stem(clean_token), smooth_eng)),
                'last_name_prob' : census_last_dict.get(clean_token, smooth_census),
                'parent_name' : np.sum([census_dict.get(normalize_token(tok.string), smooth_census) for tok in doc[j].ancestors]),
                'child_name' : np.sum([census_dict.get(normalize_token(tok.string), smooth_census) for tok in doc[j].ancestors]),
                'is_cap' : is_capital[j],
                'is_ent' : len([e for e in persons if e.start >= j and e.end <= j]) > 0,
                'is_possessive' : is_possessive[j],
                'prev_possessive' : j > 0 and is_possessive[j - 1],
                'prev_sep' : j > 0 and is_seperator[j - 1],
                'next_sep' : j < len(doc) - 1 and is_seperator[j + 1],
                'prev_and' : j > 0 and is_and[j - 1],
                'next_and' : j < len(doc) - 1 and is_and[j + 1]
                }
        token_features.append(feature_dict)
    return token_features

# Aggregate token features for an entire dataset
def add_token_features(data, token_model, english_dict, census_dict, census_last_dict, token_counter, threshold = 0.5):
    # Score tokens for each relevant voter response
    data['name_prob1'] = 0.0
    data['name_prob2'] = 0.0
    data['name_prob3'] = 0.0
    data['names_extract'] = ""
    for i, row in data.iterrows():
        if (cleanString(row['voterFinal']) == "" or row['voterFinal'] is None) and \
            (cleanString(row['voterPost']) == "" or row['voterPost'] is None):
            continue
        X_tokens_row = pd.DataFrame(
            get_token_features(row['voterFinal'], row['tripleMessage'], english_dict, census_dict, census_last_dict, token_counter) + \
            get_token_features(row['voterPost'], row['tripleMessage'], english_dict, census_dict, census_last_dict, token_counter)
            ).values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)
        top3_tokens = sorted(y_pred[:,1])[::-1][0:3]
        data.loc[i, 'name_prob1'] = top3_tokens[0]
        if len(top3_tokens) > 1:
            data.loc[i, 'name_prob2'] = top3_tokens[1]
        if len(top3_tokens) > 2:
            data.loc[i, 'name_prob3'] = top3_tokens[2]

        # Get Tokens
        doc = get_doc(row['voterFinal'])
        post_doc = get_doc(row['voterPost'])
        clean_tokens = [normalize_token(t.string) for t in doc] + [normalize_token(t.string) for t in post_doc]
        clean_tokens = [t for t in clean_tokens if not t == ""]

        full_response = row['voterResponse'] + ' ' + row['voterFinal'] + ' ' + row['voterPost']
        data.loc[i, 'names_extract'] = extract_good_tokens(clean_tokens, row['tripleMessage'], y_pred, full_response, threshold)
    return data

def featurize_conversation(data, response_vectorizer, final_vectorizer, post_vectorizer):
    # Voter Response
    X_response = response_vectorizer.transform(data['voterResponse'])

    # Voter Final
    X_final = final_vectorizer.transform(data['voterFinal'])

    # Voter Post
    X_post = post_vectorizer.transform(data['voterPost'])

    # Peripheral Features
    X_features = data[['noResponse', 'negResponse', 'posResponse', 'affirmResponse', 'finalAffirmResponse', 'name_prob1', 'name_prob2', 'name_prob3', 'num_tokens']].values * 1

    # Combine features
    X = hstack((X_response, X_final, X_post, X_features.astype('float')))

    return X
    
def main(args):

    # Set home directory
    home = Path(args.home_folder)
    THRESHOLD = args.threshold_name_prob
    DATA_FILE = Path(home, "Input_Data", args.input_data_filename)

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

    # Aggregated Message Data
    data = pd.read_csv(DATA_FILE, encoding='latin1')

    print("Cleaning and Featurizing...")

    # Fix NA Values
    data.loc[data.voterResponse.isnull(), 'voterResponse'] = ""
    data.loc[data.voterFinal.isnull(), 'voterFinal'] = ""
    data.loc[data.voterPost.isnull(), 'voterPost'] = ""

    # Number of tokens in final response
    data['num_tokens'] = data.voterFinal.str.count(" ") + ~(data.voterFinal == "")

    # Build Token Features
    data = add_token_features(data, token_model, english_dict, census_dict, census_last_dict, token_counter, threshold = THRESHOLD)

    # Build Features
    X = featurize_conversation(data, response_vectorizer, final_vectorizer, post_vectorizer)

    print("Annotating with Predictions...")

    # Add Predictions
    data['tripler_probability'] = model_tripler.predict_proba(X)[:, 1]
    data['name_provided_probability'] = model_name.predict_proba(X)[:, 1]
    data['optout_probability'] = model_opt.predict_proba(X)[:, 1]
    data['wrongnumber_probability'] = model_wrongnumber.predict_proba(X)[:, 1]

    # Eliminate names when its not likely that any are valid
    LOWER_BOUND = .4 #THRESHOLD
    UPPER_BOUND = .75 #(1 - THRESHOLD)
    MID_BOUND = .5
    triplers = data.loc[
            (data.tripler_probability > UPPER_BOUND) &
            ((data.name_provided_probability > UPPER_BOUND) | (data.name_provided_probability < LOWER_BOUND)) &
            ((data.optout_probability > UPPER_BOUND) | (data.optout_probability < LOWER_BOUND)) &
            ((data.name_prob1 > UPPER_BOUND) | (data.name_prob1 < LOWER_BOUND)) &
            ((data.name_prob2 > UPPER_BOUND) | (data.name_prob2 < LOWER_BOUND)) &
            ((data.name_prob3 > UPPER_BOUND) | (data.name_prob3 < LOWER_BOUND))
            ]
    triplers['is_tripler'] = 'yes'
    triplers.loc[triplers.name_provided_probability < UPPER_BOUND, 'names_extract'] = ''
    triplers['opted_out'] = np.where(triplers.optout_probability < UPPER_BOUND, 'no', 'yes')
    triplers['wrong_number'] = np.where(triplers.wrongnumber_probability < UPPER_BOUND, 'no', 'yes')
    triplers = triplers[['ConversationId', 'contact_phone', 
                         'is_tripler', 'opted_out', 'wrong_number', 'names_extract']]

    review = data.loc[
            ((data.tripler_probability < UPPER_BOUND) & (data.tripler_probability > LOWER_BOUND)) |
            ((data.name_provided_probability < UPPER_BOUND) & (data.name_provided_probability > LOWER_BOUND)) |
            ((data.optout_probability < UPPER_BOUND) & (data.optout_probability > LOWER_BOUND)) |
            ((data.name_prob1 < UPPER_BOUND) & (data.name_prob1 > LOWER_BOUND)) |
            ((data.name_prob2 < UPPER_BOUND) & (data.name_prob2 > LOWER_BOUND)) |
            ((data.name_prob3 < UPPER_BOUND) & (data.name_prob3 > LOWER_BOUND))
            ]
    review['is_tripler'] = np.where(review.tripler_probability < MID_BOUND, 'no', 'yes')
    review.loc[review.name_provided_probability < MID_BOUND, 'names_extract'] = ''
    review['opted_out'] = np.where(review.optout_probability < MID_BOUND, 'no', 'yes')
    review['wrong_number'] = np.where(review.wrongnumber_probability < MID_BOUND, 'no', 'yes')
    review = review[['ConversationId', 'contact_phone', 
                     'voterResponse', 'voterFinal', 'voterPost',
                     'is_tripler', 'opted_out', 'wrong_number', 'names_extract']]
    
    # Write out annotated file
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
        "-o", "--output_filename", help="File name to dump output", type=str, required=False, default='sms_triplers_annotated.csv'
    )
    PARSER.add_argument(
        "-m", "--manual_review_filename", help="File name to dump output", type=str, required=False, default='sms_manual_review.csv'
    )
    PARSER.add_argument(
        "-t", "--threshold_name_prob", help="Threshold probability to use when reporting possible extracted names", type=float, required=False, default=0.25
    )
    main(PARSER.parse_args())