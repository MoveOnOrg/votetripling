#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:24:27 2020

@author: alutes
"""
import argparse
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

def clean_labeled_names(names, affixes = AFFIXES):
    namesClean = re.sub(affixes, "", cleanString(names))
    namesClean = re.sub('\\s+', ' ', namesClean).strip()
    name_tokens = namesClean.split(' ')
    return '|'.join(present_tokens(name_tokens, names))

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
        data.loc[i, 'clean_names'] = clean_labeled_names(names)

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
        "-o", "--output_file", help="File name to dump output", type=str, required=False, default="labeled_names_cleaned.csv"
    )
    main(PARSER.parse_args())