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
POSSESSIVES = "\\b(my|his|her|their|our)\\b"
RELATIONSHIPS = "\\b((step|grand)[- ]?)?(house|kid|aunt|uncle|niece|nephew|partner|boss[a-z]+|sibling|brother|sister|son|daughter|children|child|kid|parent|mom|mother|dad|father|friend|family|cowor[a-z]+|colleague|church|pastor|priest|[a-z]*mate|husband|wife|spouse|fiance[e]*|girlfriend|boyfriend|neighbor|neighborhood|inlaw)[s]?\\b"
EXCLUDE = "\\b(your|everybody|everyone|mitch|kamala|joe|biden|member[s]*|trump|eric|tiffany|donald|melania|ivanka|idk|ty|yw|yay|oops|ooops|yes[a-z]+|ah|a|i|ill|o|y|lol|jr|sr|sir|dr|mr|mrs|ms|dr|dude|ditto|tmi|jk|rofl)\\b"
EXCLUDE_PRIOR = "\\b(im|vote for|my name is|this is|who is|this isnt|not|support|volunteer for)\\b"
NEW_LINE_REG = "\\n|\n|\\\\n"
NAME_SEPARATORS = "\\band\\b|&|\\.|,|\\n|\n|\\\\n"
NAME_AFFIXES = "\\b(mr|mrs|ms|dr|jr|sr|capt|sir|esq)\\b"
EXCLUDE_NAMES = "\\b(trump|idk|not|given|none|won't|wouldn't|no|names|hasn't|provided|say|na|none|can't)\\b"


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
    voterResponseClean = re.sub(NEW_LINE_REG, " ", voterResponse)
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
# Functions for cleaning and presenting labeled names
################################

def clean_labeled_name_string(name_string, affixes = NAME_AFFIXES):
    replaceSpecials = re.sub("(in law|in-law)[s]*", "inlaws", name_string)
    camelCleaned = re.sub("([a-z][a-z]+)([A-Z])", "\\1 \\2", replaceSpecials)
    noAffixes = re.sub(affixes, "", camelCleaned)
    noParen = re.sub("\\(.*\\)", "", noAffixes)
    return noParen

def clean_labeled_names(names, response = None, 
                        seperators = NAME_SEPARATORS, 
                        excluded = EXCLUDE_NAMES, 
                        relationships = RELATIONSHIPS,
                        affixes = re.compile(NAME_AFFIXES, re.I)):
    
    # Split the name by any and all known delimiters
    names_clean = clean_labeled_name_string(names)
    names_split = [t.strip() for t in re.split(seperators, names_clean) if not t.strip() == ""]
    
    # If we weren't able to split anything and its a 3 token response then just split by spaces
    if len(names_split) == 1 or (len(names_split) == 2 and re.search("(\\band\\b|&)", names_clean) is not None):
        names_split_spaces = [t.strip() for t in re.split(seperators+"| ", names_clean) if not t.strip() == ""]
        if len(names_split_spaces) == 3:
            names_split = names_split_spaces
    
     # Clean up the raw response, in which we will search for the tokens, if provided
    if response:
        response = clean_labeled_name_string(response)
        response_tokens = response.split(' ')
    else:
        response_tokens = []
    
    # Clean up each name section
    names_final = []
    for name in names_split:
        # Exclude bad names
        if re.match(excluded, name.lower()) is not None:
            continue
        
        # Eliminate affixes
        name = re.sub(affixes, "", name).strip()
        
        # Singular names we will do some work to ensure they are correct
        # (Any long multi-token strings we will just assume they are correct)
        if len(name.split(" ")) == 1:
                        
            # For relationships, add the appropriate 'your'
            if re.match(relationships, name.lower()) is not None:
                name = 'your ' + name.lower()   
            else:
                # If it isn't found in the response, do spell check
                if len(response_tokens) > 0 and not name in response_tokens:
                    best_match_token = get_best_match_token(name, response_tokens)
                    if best_match_token:
                        name = best_match_token

                # For initials, uppercase the whole thing
                if len(name) < 3:
                    name = name.upper()
                # For names, capitalize
                else:
                    name = name.capitalize()
        if not name in names_final:
            names_final.append(name)
            
    # Present the tokens
    return stringify_tokens(names_final, dedupe = False)

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


def clean_labeled_names_modeling(names, response = None, triple_message = None, affixes = AFFIXES):
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
                relations = RELATIONSHIPS,
                is_van_text = False):
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
            pos_match = re.search("\\b(his|her|their|step|[a-z]+'s) %s"%token, response)
            if pos_match and not is_van_text:
                token = pos_match.group()
            else:
                token = "your " + token
        elif re.match(possessive, token) is not None or \
             re.search("%s's"%token, response) is not None or \
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
def stringify_tokens(good_tokens, dedupe = True):
    if dedupe:
        name_tokens = list(set(good_tokens))
    else:
        name_tokens = good_tokens
        
    if len(name_tokens) < 1:
        return ''
    if len(name_tokens) > 1:
        name_tokens[len(name_tokens) - 1] = 'and ' + name_tokens[len(name_tokens) - 1]
       
    sep = ', '
    if len(name_tokens) == 2:
        sep = ' '
    return sep.join(name_tokens)

# Extract the tokens with high enough probability    
def extract_good_tokens(candidate_tokens, 
                        triple_message,
                        y_pred, 
                        response,
                        threshold = .5,
                        is_van_text = False):
    good_tokens = [t for j, t in enumerate(candidate_tokens) if y_pred[j, 1] > threshold]
    return present_tokens(good_tokens, response, triple_message, is_van_text = is_van_text)


################################
# Featurizing Functions
################################
        
def featurize_raw_token_position(j, clean_tokens, is_possessive, is_sep, is_and):    
    
    prefix_dict = {
            'prev' : j - 1,
            'next' : j + 1
            }

    feature_name_dict = {
            'token' : clean_tokens,
            'posessive' : is_possessive,
            'sep' : is_sep,
            'and' : is_and
            }
    # Build all token features
    feature_dict = {
            'position' : j
            }
    
    for feature_name in feature_name_dict:
        for position_name in prefix_dict:
            feature_name_full = feature_name + '_' + position_name
            position = prefix_dict[position_name]
            feature_list = feature_name_dict[feature_name]
            
            if 'token' in feature_name:
                feature_dict[feature_name_full] = ''
            else:
                feature_dict[feature_name_full] = False                
            
            # If this is a valid token position, try the feature
            if position >= 0 and position < len(is_possessive):
                feature_dict[feature_name_full] = feature_list[position]
    return feature_dict

def featurize_wordlike_token(raw_position,
                             candidate_position,
                             persons,
                             doc,
                             clean_tokens,
                             is_relationship,
                             english_dict,
                             census_dict,
                             census_last_dict,
                             token_counter,
                             smooth_census = 0,
                             smooth_eng = 1):
    raw_token = doc[raw_position].string
    clean_token = clean_tokens[raw_position]
    stemmed_token = stemmer.stem(clean_token)
    
    # Build all token features
    feature_dict = {
            'token_length' : len(clean_token),
            'candidate_position' : candidate_position,
            'relationship' : is_relationship[raw_position],
            'eng_prob' : english_dict.get(clean_token, smooth_eng),
            'name_prob' : census_dict.get(clean_token, smooth_census),
            'corpus_prob' : np.log(token_counter.get(stemmed_token, smooth_eng)),
            'last_name_prob' : census_last_dict.get(clean_token, smooth_census),
            'parent_name' : np.sum([census_dict.get(normalize_token(tok.string), smooth_census) for tok in doc[raw_position].ancestors]) > 0,
            'child_name' : np.sum([census_dict.get(normalize_token(tok.string), smooth_census) for tok in doc[raw_position].ancestors]) > 0,
            'is_cap' : re.match('[A-Z]', raw_token) is not None,
            'is_ent' : len([e for e in persons if e.start >= raw_position and e.end <= raw_position]) > 0
            }
    return feature_dict

# Get features for a single token
def get_token_features(voterResponse, 
                       tripleMessage,
                       van_token_vectorizer,
                       model_token_bow,
                       english_dict,
                       census_dict,
                       census_last_dict,
                       token_counter,
                       is_post_response = False,
                       is_van_response = False,
                       and_reg = "\\band\\b|\\by\\b|\\&|\\by\\b",
                       sep_reg = "\\||,|-|\\.",
                       relationship_reg = RELATIONSHIPS,
                       possessive_reg = POSSESSIVES,
                       excludelist_reg = EXCLUDE,
                       exclude_prior_reg = EXCLUDE_PRIOR,
                       name_threshold = 8
                       ):
    # Parse using spacy
    triple_message_tokens = cleanString(tripleMessage).split(" ")
    doc = get_doc(voterResponse)
    clean_tokens = [normalize_token(t.string) for t in doc]
    cleaned_string = ' '.join(clean_tokens)
    is_seperator = [re.match(sep_reg, t.string.lower()) is not None for t in doc]
    is_and = [re.match(and_reg, t.string.lower()) is not None for t in doc]
    is_possessive = [re.match(possessive_reg, t) is not None for t in clean_tokens]
    is_exclude = [re.match(excludelist_reg, t) is not None for t in clean_tokens]
    is_wordlike = [not t == "" for t in clean_tokens]
    is_relationship = [re.match(relationship_reg, t) is not None for t in clean_tokens]
    persons = list(doc.ents)
    
    # Reduce to name candidates
    candidate_token_positions = [i for i, t in enumerate(clean_tokens) \
                        # must be a wordlike, non "And" tooken
                        if is_wordlike[i] and not is_and[i] \
                        # must be longer than 1 letter
                        and len(t) > 1 \
                        # not an excluded phrase
                        and not is_exclude[i] \
                        # not proceeded by disqualifying phrase
                        and re.match(exclude_prior_reg + ' ' + clean_tokens[i], cleaned_string) is None \
                        # not a possesive word
                        and not is_possessive[i] \
                        # not in the solicitation message (unless its a known relationship word)
                        and (is_relationship[i] or not t in triple_message_tokens) ]
    candidate_tokens = [clean_tokens[i] for i in candidate_token_positions]

    # Featurize Each token
    all_token_features = []
    for candidate_position, raw_position in enumerate(candidate_token_positions):
        # Calculate features for adjacent raw tokens
        position_features = featurize_raw_token_position(
                raw_position, 
                clean_tokens = clean_tokens,
                is_possessive = is_possessive, 
                is_sep = is_seperator, 
                is_and = is_and
                )
        
        # Lexicon Features
        X_prev = van_token_vectorizer.transform([position_features['token_prev']])
        X_token = van_token_vectorizer.transform([clean_tokens[raw_position]])
        X_next = van_token_vectorizer.transform([position_features['token_next']])
        X_bow_row = hstack((X_prev, X_token, X_next))
        position_features['lexicon_prediction'] = model_token_bow.predict_proba(X_bow_row)[0,1]

        # Calculate features for this token
        token_features = featurize_wordlike_token(
               raw_position,
               candidate_position,
               persons,
               doc,
               clean_tokens,
               is_relationship,
               english_dict,
               census_dict,
               census_last_dict,
               token_counter
               )
        token_features.update(position_features)
        all_token_features.append(token_features)
        
    # Comment Level Features
    total_tokens = len(doc)
    word_tokens = sum(is_wordlike)
    and_tokens = sum(is_and)
    seperators = sum(is_seperator)
    comment_dict = {
            'postReponse' : is_post_response,
            'vanResponse' : is_van_response,
            'candidate_length' : len(candidate_token_positions),
            'num_tokens' : total_tokens,
            'word_tokens' : word_tokens,
            'relation_tokens' : sum(is_relationship),
            'name_tokens' : np.sum([tf['name_prob'] > name_threshold for tf in all_token_features]),
            'and_tokens' : and_tokens,
            'seperators' : seperators,
            'other_tokens' : total_tokens - word_tokens - seperators
            }

    # Which features from adjacent candidates may be relevant?
    adjacent_features = ['token_length', 'relationship', 'eng_prob', 'name_prob', 'corpus_prob', 'last_name_prob', 'is_cap']

    # Add features from adjacent tokens
    for i, tf in enumerate(all_token_features):
        
        # Add comment level features
        tf.update(comment_dict)
        
        
        for feature in adjacent_features:
            # Add previous token if applicable
            prev_feature_name = feature + '_prev'
            if i - 1 < 0:
                tf[prev_feature_name] = 0
            else:
                tf[prev_feature_name] = all_token_features[i - 1][feature]

            # Add previous token if applicable
            next_feature_name = feature + '_next'
            if i + 1 >= len(all_token_features):
                tf[next_feature_name] = 0
            else:
                tf[next_feature_name] = all_token_features[i + 1][feature]

    return candidate_tokens, all_token_features

# Aggregate token features for an entire dataset
def add_token_features(data, 
                       van_token_vectorizer, model_token_bow,
                       token_model, Features,
                       english_dict, census_dict, census_last_dict, 
                       token_counter, 
                       LOWER_BOUND = .4,
                       UPPER_BOUND = .75,
                       print_every = 1000):
    # Score tokens for each relevant voter response
    data['name_prob1'] = 0.0
    data['name_prob2'] = 0.0
    data['name_prob3'] = 0.0
    data['manual_review'] = False
    data['names_extract'] = ""
    for i, row in data.iterrows():
        if (cleanString(row['voterfinal'], ) == "" or row['voterfinal'] is None) and \
            (cleanString(row['voterpost']) == "" or row['voterpost'] is None):
            continue

        if i%print_every == 0:
            print("processing row %s"%i)

        # Featurize available tokens 
        finalCandidates, finalFeatures = get_token_features(row['voterfinal'], row['triplemessage'],
                                                            van_token_vectorizer, model_token_bow,
                                                            english_dict, census_dict, 
                                                            census_last_dict, token_counter)
        postCandidates, postFeatures = get_token_features(row['voterpost'], row['triplemessage'], 
                                                            van_token_vectorizer, model_token_bow,
                                                          english_dict, census_dict, 
                                                          census_last_dict, token_counter, 
                                                          is_post_response = True)
        candidates = finalCandidates + postCandidates
        
        if len(candidates) <= 0:
            continue
        # Predict probability of each being a name
        X_tokens_row = pd.DataFrame(finalFeatures + postFeatures)[Features].values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)

        # Do we have an uncertain token?
        if ((y_pred[:,1] > LOWER_BOUND) & (y_pred[:,1] < UPPER_BOUND)).sum() > 0:
            data.loc[i, 'manual_review'] = True

        # Store probabilities for the Top 3
        top3_tokens = sorted(y_pred[:,1])[::-1][0:3]
        data.loc[i, 'name_prob1'] = top3_tokens[0]
        if len(top3_tokens) > 1:
            data.loc[i, 'name_prob2'] = top3_tokens[1]
        if len(top3_tokens) > 2:
            data.loc[i, 'name_prob3'] = top3_tokens[2]

        full_response = row['voterresponse'] + ' ' + row['voterfinal'] + ' ' + row['voterpost']
        data.loc[i, 'names_extract'] = extract_good_tokens(
                candidate_tokens = candidates,
                triple_message = row['triplemessage'], 
                y_pred = y_pred, 
                response = full_response, 
                threshold = LOWER_BOUND
                )
    return data

# Aggregate token features for an entire dataset
def add_token_features_van(data, van_token_vectorizer, model_token_bow,
                           token_model, Features,
                           english_dict, census_dict, 
                           census_last_dict, token_counter, 
                           LOWER_BOUND = .4,
                           UPPER_BOUND = .75,
                           print_every = 1000):
    # Score tokens for each relevant voter response
    data['name_prob1'] = 0.0
    data['name_prob2'] = 0.0
    data['name_prob3'] = 0.0
    data['names_extract'] = ""
    for i, row in data.iterrows():
        if (cleanString(row['notetext'], ) == "" or row['notetext'] is None):
            continue

        if i%print_every == 0:
            print("processing row %s"%i)

        # Featurize available tokens 
        candidates, features = get_token_features(row['notetext'], row['contactname'], 
                                                  van_token_vectorizer, model_token_bow,
                                                  english_dict, census_dict, 
                                                  census_last_dict, token_counter,
                                                  is_van_response = True)
        
        if len(candidates) <= 0:
            continue

        # Predict probability of each being a name
        X_tokens_row = pd.DataFrame(features)[Features].values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)

        # Do we have an uncertain token?
        if ((y_pred[:,1] > LOWER_BOUND) & (y_pred[:,1] < UPPER_BOUND)).sum() > 0:
            data.loc[i, 'manual_review'] = True

        # Store probabilities for the Top 3
        top3_tokens = sorted(y_pred[:,1])[::-1][0:3]
        data.loc[i, 'name_prob1'] = top3_tokens[0]
        if len(top3_tokens) > 1:
            data.loc[i, 'name_prob2'] = top3_tokens[1]
        if len(top3_tokens) > 2:
            data.loc[i, 'name_prob3'] = top3_tokens[2]

        data.loc[i, 'names_extract'] = extract_good_tokens(
                candidate_tokens = candidates,
                triple_message = row['contactname'], 
                y_pred = y_pred, 
                response = row['notetext'], 
                threshold = LOWER_BOUND,
                is_van_text = True
                )
    return data

def featurize_conversation_van(data, van_vectorizer):
    # Voter Response
    X_response = van_vectorizer.transform(data['notetext'])

    # Peripheral Features
    X_features = data[['name_prob1', 'name_prob2', 'name_prob3', 'num_tokens']].values * 1

    # Combine features
    X = hstack((X_response, X_features.astype('float')))

    return X

def featurize_conversation(data, response_vectorizer, final_vectorizer, post_vectorizer):
    # Voter Response
    X_response = response_vectorizer.transform(data['voterresponse'])

    # Voter Final
    X_final = final_vectorizer.transform(data['voterfinal'])

    # Voter Post
    X_post = post_vectorizer.transform(data['voterpost'])

    # Peripheral Features
    X_features = data[['noresponse', 'negresponse', 'posresponse', 
                       'affirmresponse', 'finalaffirmresponse', 
                       'name_prob1', 'name_prob2', 'name_prob3', 
                       'num_tokens_response',
                       'num_tokens_final',
                       'num_tokens_post']].values * 1

    # Combine features
    X = hstack((X_response, X_final, X_post, X_features.astype('float')))

    return X