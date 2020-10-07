#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:49:04 2020

@author: alutes
"""

import argparse
import Levenshtein
from pathlib import Path
import re
import pandas as pd
AFFIXES = "\\b(mr|mrs|ms|dr|jr|sr|your|her|his|our|their|in|you)\\b"
POSSESSIVES = "\\b(my|his|her|their|our|step)\\b"
RELATIONSHIPS = "\\b((step|grand)[- ]?)?(house|kid|aunt|uncle|partner|ma|pa|boss[a-z]+|follower|sibling|brother|sister|son|daughter|children|child|kid|parent|mom|mother|dad|father|friend|family|cowor[a-z]+|colleague|church|pastor|priest|[a-z]*mate|husband|wife|spouse|girlfriend|boyfriend|neighbo[a-z]+|in[ -]?law)[s]*\\b"
EXCLUDE = "\\b(member|trump|donald|melania|ivanka|idk|ty|yw|yay|oops|ooops|yes[a-z]+|ah|ill|o|y|lol|jr|sr|sir|dr|mr|mrs|ms|dr|dude|ditto|tmi|jk|rofl)\\b"
EXCLUDE_PRIOR = "\\b(im|vote for|my name is|this is|who is|this isnt|not|support)\\b"
NEW_LINE_REG = "\\n|\n|\\\\n"

def cleanString(string, exclude_reg = '\\&|\\band\\b|\\bmy\\b'):
    noNewLine = re.sub(NEW_LINE_REG, " ", string)
    camelCleaned = re.sub("([a-z][a-z]+)([A-Z])", "\\1 \\2", noNewLine)
    noAnd = re.sub(exclude_reg, ' ', camelCleaned.lower())
    noChar = re.sub('[^a-z ]', ' ', noAnd)
    return re.sub('\\s+', ' ', noChar).strip()

def get_best_match_token(t, tokens_to_match):
    best_match_token = None
    jw_best = 0
    for s in tokens_to_match:
        dist = Levenshtein.distance(s,t)
        jw = Levenshtein.jaro_winkler(s,t)
        if dist <= 2 and len(t) > 3:
            if jw > jw_best:
                jw_best = jw
                best_match_token = s
    return best_match_token

def clean_labeled_names(names, response, affixes = AFFIXES):
    namesClean = re.sub(affixes, "", cleanString(names))
    namesClean = re.sub('\\s+', ' ', namesClean).strip()
    name_tokens = namesClean.split(' ')
    response = cleanString(response)
    response_tokens = response.split(' ')
    
    clean_name_tokens = []
    for t in name_tokens:
        if t not in response_tokens:            
            best_match_token = get_best_match_token(t, response_tokens)
            if best_match_token:
                clean_name_tokens.append(best_match_token)
        else:
            clean_name_tokens.append(t)
    return '|'.join(present_tokens(clean_name_tokens, response))

def present_tokens(clean_tokens, 
                response,
                excluded = EXCLUDE,
                possessive = POSSESSIVES,
                relations = RELATIONSHIPS):
    good_tokens = []
    for j, token in enumerate(clean_tokens):
        if re.match(possessive, token) is None and re.match(excluded, token) is None:
            # For relationships, look for the proper modifier
            if re.match(relations, token):
                pos_match = re.search("\\b(his|her|their|step) %s"%token, response)
                if pos_match:
                    token = pos_match.group()
                else:
                    token = "your " + token
            # For names, capitalize
            else:
                token = token.capitalize()
            good_tokens.append(token)
    return set(good_tokens)

def main(args):
    
    # Set home directory
    home = Path(args.home_folder)
    DATA_FILE = Path(home, args.data_file)

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
        data.loc[i, 'clean_names'] = clean_labeled_names(names, response)

    # Write out annotated file
    data.to_csv(Path(home, args.output_file), index = False)

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