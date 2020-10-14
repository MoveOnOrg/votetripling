#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:24:28 2020

@author: alutes
"""
import re
import Levenshtein
import spacy
from nltk import SnowballStemmer
from scipy.sparse import hstack
import pandas as pd
import numpy as np
from pathlib import Path

################################
# Define Global Objects
################################

stemmer = SnowballStemmer('english')
nlp = spacy.load('en')
AFFIXES = "\\b(mr|mrs|ms|dr|jr|sr|your|her|his|our|their|in|you)\\b"
POSSESSIVES = "\\b(my|his|her|their|our|step)\\b"
RELATIONSHIPS = "\\b((step|grand)[- ]?)?(house|kid|aunt|uncle|niece|nephew|partner|ma|pa|boss[a-z]+|follower|sibling|brother|sister|son|daughter|children|child|kid|parent|mom|mother|dad|father|friend|family|cowor[a-z]+|colleague|church|pastor|priest|[a-z]*mate|husband|wife|spouse|fiance[e]*|girlfriend|boyfriend|neighbo[a-z]+|in[ -]?law)[s]*\\b"
EXCLUDE = "\\b(everyone|kamala|biden|member[s]*|trump|eric|tiffany|donald|melania|ivanka|idk|ty|yw|yay|oops|ooops|yes[a-z]+|ah|ill|o|y|lol|jr|sr|sir|dr|mr|mrs|ms|dr|dude|ditto|tmi|jk|rofl)\\b"
EXCLUDE_PRIOR = "\\b(im|vote for|my name is|this is|who is|this isnt|not|support|volunteer for)\\b"
NEW_LINE_REG = "\\n|\n|\\\\n"

################################
# Data Loading
################################

def load_flat_file(home, filename):
    DATA_FILE = Path(home, "Input_Data", filename)
    data = pd.read_csv(DATA_FILE, encoding="latin1")
    data.columns = [c.lower() for c in data.columns]
    return data

def load_civis(tablename, db = "Vote Tripling"):
    import civis
    data = civis.io.read_civis(table=tablename, 
                               database=db, 
                               use_pandas=True)
    return data

def export_civis(df, tablename, db = "Vote Tripling"):
    import civis
    civis.io.dataframe_to_civis(df, 
                                database=db, 
                                table=tablename)
    
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


def get_list(lst, index):
    if index < 0 or index >= len(lst):
        return ''
    else:
        return lst[index]

def cleanString(string, splitCamel = True, exclude_reg = '\\&|\\band\\b|\\bmy\\b'):
    replaceSpecials = re.sub("(in law|in-law)[s]*", "inlaws", string)
    noNewLine = re.sub(NEW_LINE_REG, " ", replaceSpecials)
    if splitCamel:
        camelCleaned = re.sub("([a-z][a-z]+)([A-Z])", "\\1 \\2", noNewLine)
    else:
        camelCleaned = noNewLine
    noAnd = re.sub(exclude_reg, ' ', camelCleaned.lower())
    noChar = re.sub('[^a-z ]', ' ', noAnd)
    return re.sub('\\s+', ' ', noChar).strip()

################################
# Functions for cleaning and presenting names
################################

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


def clean_labeled_names(names, response = None, triple_message = None, affixes = AFFIXES):
    namesClean = re.sub(affixes, "", cleanString(names))
    namesClean = re.sub('\\s+', ' ', namesClean).strip()
    name_tokens = namesClean.split(' ')
    
    # Clean up the raw response, in which we will search for the tokens, if provided
    if response:
        response = cleanString(response)
        response_tokens = response.split(' ')
    else:
        response_tokens = []
    
    clean_name_tokens = []
    for t in name_tokens:
        if t not in response_tokens and len(response_tokens) > 0:            
            best_match_token = get_best_match_token(t, response_tokens)
            if best_match_token:
                clean_name_tokens.append(best_match_token)
        else:
            clean_name_tokens.append(t)
    return present_tokens(clean_name_tokens, response, triple_message)

# Take a set of putative extracted or provided name tokens and clean them up
def present_tokens(clean_tokens, 
                response,
                triple_message = None,
                excluded = EXCLUDE,
                possessive = POSSESSIVES,
                relations = RELATIONSHIPS):
    # If Provided, make sure to exclude any names in the initial triple message 
    # as they are either the target or a politician
    if triple_message:
        triple_tokens = cleanString(triple_message).split(" ")
    else:
        triple_tokens = []
    
    # Compile a list of acceptable tokens
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
        # For initials, uppercase the whole thing
        elif len(token) < 3:
            token = token.upper()
        # For names, capitalize
        else:
            token = token.capitalize()
        good_tokens.append(token)
        
    # Present the tokens
    return stringify_tokens(good_tokens)

# Turn a list of tokens into one string
def stringify_tokens(good_tokens):
    name_tokens = list(set(good_tokens))
    if len(name_tokens) < 1:
        return ''
    if len(name_tokens) > 1:
        name_tokens[len(name_tokens) - 1] = 'and ' + name_tokens[len(name_tokens) - 1]
       
    sep = ', '
    if len(name_tokens) == 2:
        sep = ' '
    return sep.join(name_tokens)

# Extract the tokens with high enough probability    
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
        if (cleanString(row['voterfinal'], ) == "" or row['voterfinal'] is None) and \
            (cleanString(row['voterpost']) == "" or row['voterpost'] is None):
            continue
        X_tokens_row = pd.DataFrame(
            get_token_features(row['voterfinal'], row['triplemessage'], english_dict, census_dict, census_last_dict, token_counter) + \
            get_token_features(row['voterpost'], row['triplemessage'], english_dict, census_dict, census_last_dict, token_counter)
            ).values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)
        top3_tokens = sorted(y_pred[:,1])[::-1][0:3]
        data.loc[i, 'name_prob1'] = top3_tokens[0]
        if len(top3_tokens) > 1:
            data.loc[i, 'name_prob2'] = top3_tokens[1]
        if len(top3_tokens) > 2:
            data.loc[i, 'name_prob3'] = top3_tokens[2]

        # Get Tokens
        doc = get_doc(row['voterfinal'])
        post_doc = get_doc(row['voterpost'])
        clean_tokens = [normalize_token(t.string) for t in doc] + [normalize_token(t.string) for t in post_doc]
        clean_tokens = [t for t in clean_tokens if not t == ""]

        full_response = row['voterresponse'] + ' ' + row['voterfinal'] + ' ' + row['voterpost']
        data.loc[i, 'names_extract'] = extract_good_tokens(
                clean_tokens = clean_tokens, 
                triple_message = row['triplemessage'], 
                y_pred = y_pred, 
                response = full_response, 
                threshold = threshold
                )
    return data

def featurize_conversation(data, response_vectorizer, final_vectorizer, post_vectorizer):
    # Voter Response
    X_response = response_vectorizer.transform(data['voterresponse'])

    # Voter Final
    X_final = final_vectorizer.transform(data['voterfinal'])

    # Voter Post
    X_post = post_vectorizer.transform(data['voterpost'])

    # Peripheral Features
    X_features = data[['noresponse', 'negresponse', 'posresponse', 'affirmresponse', 'finalaffirmresponse', 'name_prob1', 'name_prob2', 'name_prob3', 'num_tokens']].values * 1

    # Combine features
    X = hstack((X_response, X_final, X_post, X_features.astype('float')))

    return X