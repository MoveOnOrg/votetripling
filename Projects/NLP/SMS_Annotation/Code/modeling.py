#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:36:16 2020

@author: alutes
"""
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
import matplotlib.ticker as mtick
from itertools import dropwhile
from scipy.sparse import hstack
from utilities import get_token_features, stemmer, normalize_token, clean_labeled_names, extract_good_tokens, cleanString

def get_response_names(namesClean,
                       poss = "\\b(your|my|his|her|their|our|step|and|[0-9])\\b"):
    tokens = namesClean.lower().replace(",", "").split(" ")
    return [t for t in tokens if re.match(poss, t) is None]

################################
# Load and Clean SMS Data
################################
    
# Load Data
home = Path("../")

# Train Data
labeled = pd.read_csv(Path(home, "Input_Data/labeled_agg.csv"), encoding='latin1')

# Fix NA values
labeled.loc[labeled.voterResponse.isnull(), 'voterResponse'] = ""
labeled.loc[labeled.voterFinal.isnull(), 'voterFinal'] = ""
labeled.loc[labeled.voterPost.isnull(), 'voterPost'] = ""
labeled.loc[labeled.names.isnull(), 'names'] = ""
    
# Eliminate names from people who moved
labeled.loc[(labeled.moved == 1) & ~(labeled.names == ""), ['names']] = ""
labeled.loc[(labeled.wrongnumber == 1) & ~(labeled.names == ""), ['names']] = ""

# How many tokens
labeled['num_tokens_response'] = labeled.voterResponse.str.count(" ") + ~(labeled.voterResponse == "")
labeled['num_tokens_final'] = labeled.voterFinal.str.count(" ") + ~(labeled.voterFinal == "")
labeled['num_tokens_post'] = labeled.voterPost.str.count(" ") + ~(labeled.voterPost == "")

################################
# Load and Clean SMS Data
################################

## Training Data ##
columns = ['voter_file_vanid', 'contactname', 'notetext', 'names_extract', 'source']

# Primary File
van1 = pd.read_csv(Path(home, "Input_Data/van_manualreview.csv"), encoding='latin1')
van1 = van1[['voter_file_vanid', 'contactname', 'notetext']].drop_duplicates()
van1 = van1.fillna('')
van1['source'] = 'main'

# Group together all notes
van1['notetext'] = van1.groupby(['voter_file_vanid', 'contactname', 'source'])['notetext'].transform(lambda x: ' '.join(x))
van1 = van1.drop_duplicates()

# Add on labels
van1_labels = pd.read_csv(Path(home, "Input_Data", "van_manualreview_labels.csv"), encoding='latin1').drop_duplicates()
van1_labels = van1_labels.fillna('')
van1 = pd.merge(van1, van1_labels, on = 'voter_file_vanid', how = 'outer')
van1 = van1[columns].drop_duplicates()

# Eliminate those with multiple contradicting labels
ids = van1.loc[~(van1.names_extract == "")].voter_file_vanid.value_counts()
ids = ids.loc[ids > 1].index
van1 = van1.loc[~(van1.voter_file_vanid.isin(ids))]

# For the rest, take the best extraction available
van1 = van1.groupby(['voter_file_vanid', 'contactname', 'notetext', 'source'], 
                   as_index = False)['names_extract'].max()

# Additional Files
van2_columns = ['voter file vanid', 'firstname', 'middlename', 'lastname', 'notes', '3friends']
files = ["van_driskel.csv", "van_hd20.csv", "van_hd51.csv"]
dfs = []
for f in files:
    df = pd.read_csv(Path(home, "Input_Data", f), encoding='latin1')
    df.columns = [c.strip().lower() for c in df.columns]    
    dfs.append(df[van2_columns])
van2 = pd.concat(dfs)
van2 = van2.fillna('')
van2['voter_file_vanid'] = van2['voter file vanid']
van2['contactname'] = van2['firstname'] + ' ' + van2['middlename'] + ' ' + van2['lastname']
van2['notetext'] = van2['notes']
van2['names_extract'] = van2['3friends']
van2['source'] = 'hd'

# Combine Files
van = pd.concat([van1[columns], van2[columns]])

# Fix NA values
van.loc[van.notetext.isnull(), 'notetext'] = ""
van.loc[van.names_extract.isnull(), 'names_extract'] = ""
van.loc[van.contactname.isnull(), 'contactname'] = ""

# Eliminate generic answer of 3 friends
van = van.loc[~(van.names_extract.isin(['your 3 friends', 'your three friends']))]
van = van.loc[~(van['notetext'] == "")]
van.to_csv(Path(home, "Input_Data", "van_all.csv"))

# How Many tokens
van['num_tokens'] = van.notetext.str.count(" ") + ~(van.notetext == "")

################################
# Clean Labeled Names
################################

# Clean Name Column
labeled['namesClean'] = ''
for i, row in labeled.loc[~(labeled.names == '')].iterrows():
    names = row['names']
    response = row['voterResponse'] + ' ' + row['voterFinal'] + ' ' + row['voterPost']
    labeled.loc[i, 'namesClean'] = clean_labeled_names(names, response)
labeled = labeled.loc[~(~(labeled.names == "") & (labeled.namesClean == ""))]

van.index = np.arange(len(van))
van['namesClean'] = ''
for i, row in van.loc[~(van.names_extract == '')].iterrows():
    names = row['names_extract']
    response = row['notetext']
    van.loc[i, 'namesClean'] = clean_labeled_names(names, response)
van = van.loc[~(~(van.names_extract == "") & (van.namesClean == ""))]


# Name Data
census = pd.read_csv(Path(home, "census_first_names_all.csv"))
census_dict = {}
for i, row in census.iterrows():
    census_dict[row['name']] = np.log(row['census_count'])

# Name Data
census_last = pd.read_csv(Path(home, "census_last_names_all.csv"))
census_last_dict = {}
for i, row in census_last.iterrows():
    census_last_dict[row['name']] = np.log(row['census_count'])

# US Word Freq Data
english = pd.read_csv(Path(home, "english.csv"))
english_dict = {}
for i, row in english.iterrows():
    english_dict[row['name']] = row['freq']

# Corpus of all non-name responses
noname_corpus = np.concatenate((
        labeled.loc[~(labeled.voterResponse == ""), "voterResponse"].values,
        labeled.loc[~(labeled.voterFinal == "") & (labeled.names == ""), "voterFinal"].values,
        labeled.loc[~(labeled.voterPost == "") & (labeled.names == ""), "voterPost"].values,
        van.loc[~(van.notetext == "") & (van.names_extract == ""), "notetext"].values
        ))
token_counter = Counter()
for response in noname_corpus:
    for token in re.split("\\W+", response):
        token_counter[stemmer.stem(normalize_token(token))] += 1
for key, count in dropwhile(lambda key_count: key_count[1] >= 5, token_counter.most_common()):
    del token_counter[key]


################################
# General Token Featurization for SMS
################################

### BOW Representations for Token Model ###

# Voter Final
van_token_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=25)
X_bow_token = van_token_vectorizer.fit_transform(van.notetext.values.tolist() + \
                                                 labeled.voterPost.values.tolist() + \
                                                 labeled.voterFinal.values.tolist())

### BOW Representations for Response Models ###
    
# Voter Response
response_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=5)
X_response = response_vectorizer.fit_transform(labeled.voterResponse)

# Voter Final
final_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=5)
X_final = final_vectorizer.fit_transform(labeled.voterFinal)

# Voter Post
post_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=5)
X_post = post_vectorizer.fit_transform(labeled.voterPost)

# Peripheral Features
main_feature_names = ['noResponse', 'negResponse', 'posResponse', 
                      'affirmResponse', 'finalAffirmResponse', 
                      'name_prob1', 'name_prob2', 'name_prob3', 
                      'num_tokens_response',
                      'num_tokens_final',
                      'num_tokens_post']
X_features = labeled[main_feature_names].values * 1

# Combine features
X = hstack((X_response, X_final, X_post, X_features.astype('float')))

# Response Variables
y_tripler = pd.get_dummies(labeled.tripler.astype('category'))
y_optout = pd.get_dummies(labeled.optout.astype('category'))
y_names = 1*~(labeled.names.isnull() | (labeled.names == ""))
y_wrongnumber = pd.get_dummies(labeled.wrongnumber.astype('category'))

# Train/Test
train, test, X_train, X_test = train_test_split(np.arange(X.shape[0]), X)


################################
# Name Token Features
################################

token_features_all = []
for i, row in labeled.loc[labeled['tripler']==1].iterrows():
    responses = get_response_names(row['namesClean'])
    finalCandidates, finalFeatures = get_token_features(row['voterFinal'], row['tripleMessage'],
                                                        van_token_vectorizer, model_token_bow,
                                                        english_dict, census_dict, census_last_dict, 
                                                        token_counter)
    postCandidates, postFeatures = get_token_features(row['voterPost'], row['tripleMessage'],
                                                      van_token_vectorizer, model_token_bow,
                                                      english_dict, census_dict, census_last_dict, 
                                                      token_counter, is_post_response = True)
    new_df = pd.DataFrame(finalFeatures + postFeatures)
    new_df['token'] = finalCandidates + postCandidates
    new_df['response'] = [t in responses for t in new_df['token']]
    new_df['actual'] = row['namesClean']
    new_df['full'] = row['voterResponse'] + ' ' + row['voterFinal'] + ' ' +row['voterPost']
    token_features_all.append(new_df)


for i, row in van.iterrows():
    responses = get_response_names(row['namesClean'])
    candidates, features = get_token_features(row['notetext'], row['contactname'],
                                          van_token_vectorizer, model_token_bow,
                                          english_dict, census_dict, census_last_dict,
                                          token_counter, is_van_response = True)
    new_df = pd.DataFrame(features)
    new_df['token'] = candidates
    new_df['response'] = [t in responses for t in new_df['token']]
    new_df['actual'] = row['namesClean']
    new_df['full'] = row['notetext']
    token_features_all.append(new_df)

token_df = pd.concat(token_features_all)
token_df.index = np.arange(len(token_df))

################################
# Token Featurization
################################

# Class Weight
y_tokens = token_df['response'].values.astype(bool)
class_prob = y_tokens.mean()
class_weight = {True : 1 / class_prob, False : 1}

# Featurize Lexicon
X_prev = van_token_vectorizer.transform(token_df['token_prev'].astype(str))
X_token = van_token_vectorizer.transform(token_df['token'].astype(str))
X_next = van_token_vectorizer.transform(token_df['token_next'].astype(str))
X_bow = hstack((X_prev, X_token, X_next))

# Train/Test Split
X_bow_train, X_bow_test, train_tokens, test_tokens = train_test_split(X_bow, np.arange(len(token_df)))
y_train_tokens = y_tokens[train_tokens]
y_test_tokens = y_tokens[test_tokens]

# Train BOW Model
model_token_bow = LogisticRegressionCV(penalty='l1', solver='liblinear', class_weight = class_weight)
model_token_bow.fit(X_bow_train, y_train_tokens)

# Predictions from BOW
token_df['lexicon_prediction'] = model_token_bow.predict_proba(X_bow)[:, 1]

# Full Token Featurization
not_Features = ['response', 'token', 'actual', 'full', 'token_next', 'token_prev']
Features = [c for c in token_df.columns if not c in not_Features]
X_tokens = token_df[Features].values.astype(float)
X_train_tokens = X_tokens[train_tokens, :]
X_test_tokens = X_tokens[test_tokens, :]

################################
# Train Token Model
################################

# Train Model
token_model = RandomForestClassifier(n_estimators = 250, 
                                     class_weight = class_weight)
token_model.fit(X_train_tokens, y_train_tokens)

# Evaluate
token_df['prob'] = token_model.predict_proba(X_tokens)[:, 1]
plt.hist(token_df.loc[test_tokens,'prob'])
plt.hist(token_df.loc[train_tokens,'prob'])

# Print weird examples
token_df.loc[(token_df['prob'] > .75) & (token_df['response'] == False)][['token', 'full', 'actual', 'prob']]
token_df.loc[(token_df['prob'] < .25) & (token_df['response'] == True)][['token', 'full', 'actual', 'prob']]
token_df.loc[(token_df['relationship'] == True) & (token_df['response'] == False)][['token', 'full', 'actual', 'prob']]

################################
# Model for whether the response has names in it
################################

# Score tokens for each relevant voter response
labeled['name_prob1'] = 0.0
labeled['name_prob2'] = 0.0
labeled['name_prob3'] = 0.0
labeled['names_extract'] = ""
threshold = 0.5
for i, row in labeled.iterrows():
    if (cleanString(row['voterFinal']) == "" or row['voterFinal'] is None) and \
            (cleanString(row['voterPost']) == "" or row['voterPost'] is None):
        continue
    finalCandidates, finalFeatures = get_token_features(row['voterFinal'], row['tripleMessage'], 
                                                        van_token_vectorizer, model_token_bow,
                                                        english_dict, census_dict, census_last_dict, token_counter)
    postCandidates, postFeatures = get_token_features(row['voterPost'], row['tripleMessage'], 
                                                      van_token_vectorizer, model_token_bow,
                                                      english_dict, census_dict, census_last_dict, 
                                                      token_counter, is_post_response = True)
    candidates = finalCandidates + postCandidates
    if len(candidates) > 0:
        row_token_df = pd.DataFrame(finalFeatures + postFeatures)
        X_tokens_row = row_token_df[Features].values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)
        top3_tokens = sorted(y_pred[:,1])[::-1][0:3]
        if len(top3_tokens) > 0:
            labeled.loc[i, 'name_prob1'] = top3_tokens[0]
        if len(top3_tokens) > 1:
            labeled.loc[i, 'name_prob2'] = top3_tokens[1]
        if len(top3_tokens) > 2:
            labeled.loc[i, 'name_prob3'] = top3_tokens[2]
        labeled.loc[i, 'names_extract'] = extract_good_tokens(
                candidates, 
                row['tripleMessage'],
                y_pred, 
                row['voterFinal'] + ' ' + row['voterPost'],
                threshold
                )


# Score tokens for each relevant voter response
van['name_prob1'] = 0.0
van['name_prob2'] = 0.0
van['name_prob3'] = 0.0
van['names_model'] = ""
threshold = 0.5
for i, row in van.iterrows():
    if (cleanString(row['notetext']) == "" or row['notetext'] is None):
        continue
    candidates, features = get_token_features(
            row['notetext'], row['contactname'], 
            van_token_vectorizer, model_token_bow,
            english_dict, census_dict, census_last_dict, token_counter
            )
    if len(candidates) > 0:
        X_tokens_row = pd.DataFrame(features)[Features].values.astype(float)
        y_pred = token_model.predict_proba(X_tokens_row)
        top3_tokens = sorted(y_pred[:,1])[::-1][0:3]
        van.loc[i, 'name_prob1'] = top3_tokens[0]
        if len(top3_tokens) > 1:
            van.loc[i, 'name_prob2'] = top3_tokens[1]
        if len(top3_tokens) > 2:
            van.loc[i, 'name_prob3'] = top3_tokens[2]
            
        van.loc[i, 'names_model'] = extract_good_tokens(
                    candidates, 
                    row['contactname'],
                    y_pred, 
                    row['notetext'],
                    threshold
                    )

# Re-index to ensure no missing
labeled.index = np.arange(len(labeled))
van.index = np.arange(len(van))

################################
# General Model Training SMS
################################

# Is this a tripler?
model_tripler = LogisticRegressionCV(penalty='l1', solver='liblinear')
model_tripler.fit(X_train, y_tripler.values[train, 1])

# Did they provide a valid name?
model_name = LogisticRegressionCV(penalty='l1', solver='liblinear')
model_name.fit(X_train, y_names[train])

# Did they Opt Out?
model_opt = LogisticRegressionCV(penalty='l1', solver='liblinear')
model_opt.fit(X_train, y_optout.values[train, 1])

# Wrong Number?
model_wrongnumber = LogisticRegressionCV(penalty='l1', solver='liblinear')
model_wrongnumber.fit(X_train, y_wrongnumber.values[train, 1])


################################
# General Token Featurization for VAN
################################

# Voter Response
van_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=5)
X_van_response = van_vectorizer.fit_transform(van.notetext)

# Peripheral Features
main_feature_names = ['name_prob1', 'name_prob2', 'name_prob3', 'num_tokens']
X_van_features = van[main_feature_names].values * 1

# All Features
X_van = hstack((X_van_response, X_van_features.astype('float')))

# Response Variable
y_van_names = 1*~(van.namesClean == "")

# Train/Test
van_train, van_test, X_van_train, X_van_test = train_test_split(np.arange(X_van.shape[0]), X_van)

# Did they provide a valid name?
model_van_name = LogisticRegressionCV(penalty='l1', solver='liblinear')
model_van_name.fit(X_van_train, y_van_names[van_train])

################################
# Print Lexicons
################################

names = response_vectorizer.get_feature_names() + \
        final_vectorizer.get_feature_names() + \
        post_vectorizer.get_feature_names() + \
        main_feature_names

feature_df = pd.DataFrame({
        'tripler_weight' : model_tripler.coef_[0],
        'provided_name_weight' : model_name.coef_[0],
        'optout_weight' : model_opt.coef_[0],
        'wrongnumber_weight' : model_wrongnumber.coef_[0],
        }, index = names)
feature_df.loc[~((feature_df.tripler_weight == 0) &
                 (feature_df.provided_name_weight == 0) &
                 (feature_df.optout_weight == 0) &
                 (feature_df.wrongnumber_weight == 0))].to_csv("./model_features.csv")

    
names = van_vectorizer.get_feature_names()
feature_df = pd.DataFrame({
        'weight' : model_van_name.coef_[0][0:len(names)]
        }, index = names)
feature_df.loc[~((feature_df.weight == 0))]


token_importance = pd.DataFrame({'importance' : token_model.feature_importances_}, 
                                index = pd.DataFrame(features).columns)
token_importance.sort_values('importance')

################################
# Pickle all models and featurizers
################################

pickle_file = Path(home, "models", "annotation_models.pkl")
with open(pickle_file, "wb") as f:
    # N-Gram Featurizers
    pickle.dump(response_vectorizer, f)
    pickle.dump(final_vectorizer, f)
    pickle.dump(post_vectorizer, f)

    # Logistic Regressions
    pickle.dump(token_model, f)
    pickle.dump(model_tripler, f)
    pickle.dump(model_name, f)
    pickle.dump(model_opt, f)
    pickle.dump(model_wrongnumber, f)
    
    # Other Models
    pickle.dump(token_counter, f)
    pickle.dump(model_van_name, f)
    pickle.dump(van_vectorizer, f)
    pickle.dump(Features, f)
    pickle.dump(model_token_bow, f)
    pickle.dump(van_token_vectorizer, f)

################################
# Validation
################################

def plot_validation(actuals, score, title, 
                    lower_threshold = .2, 
                    upper_threshold = .8,
                    figsize = (6,6)):
    # Calculate validation statistics
    fpr, tpr, thresholds = roc_curve(actuals, score)
    auc = roc_auc_score(actuals, score)

    precision, recall, thresholds = precision_recall_curve(actuals, score)
    thresholds = np.concatenate((thresholds, [1]))

    # Caclulate Manual Review Stats
    lower_recall = np.min(recall[thresholds < lower_threshold])
    upper_recall = np.min(recall[thresholds < upper_threshold])

    lower_prec = np.max(precision[thresholds < lower_threshold])
    upper_prec = np.max(precision[thresholds < upper_threshold])

    upper_number_flagged = np.sum(score > upper_threshold)
    lower_number_ignored = np.sum(score < lower_threshold)
    number_reviewed = np.sum((score < upper_threshold) & (score > lower_threshold))
    lower_number_missed = np.sum((score < lower_threshold) & (actuals == 1))
    review_precision = np.sum((score < upper_threshold) & (score > lower_threshold) & (actuals == 1)) / number_reviewed
        
    # ROC Curve
    fig = plt.figure(1, figsize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(fpr, tpr)

    # Format Plot
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha = .5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    # Annotate
    plt.title(title, weight='bold', size = 15)
    ax.annotate("AUC: {}%".format(round(auc*100, 2)), xy = (.2, .8), size = 14, style='italic')    
    plt.show()

    # Precision Recall
    fig = plt.figure(1, figsize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(recall, precision)

    # Format Plot
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.grid(color='grey', linestyle='--', linewidth=1, alpha = .5)
    ax.set_xlabel('Recall', size = 12)
    ax.set_ylabel('Precision', size = 12)
    
    # Annotate
    ax.scatter([lower_recall, upper_recall],
               [lower_prec, upper_prec],
               marker = "x", color ="r")
    
    ax.annotate("Auto categorize {} records over \n {}% with {}% precision".format(
            upper_number_flagged,
            round(upper_threshold*100, 0),
            round(upper_prec*100, 0)
            ), xy = (upper_recall, upper_prec),
            va = 'bottom')

    ax.annotate("Auto ignore {} records under {}%, \n accidentally missing {} positive examples".format(
            lower_number_ignored,
            round(lower_threshold*100, 0),
            lower_number_missed
            ), xy = (lower_recall, lower_prec),
            va = 'top')

    ax.annotate("Manually review {} records with \n {}% precision".format(
            number_reviewed,
            round(review_precision*100, 0)
            ), xy = (upper_recall, lower_prec),
            ha='right', va = 'top')
    plt.title(title, weight='bold', size = 15)
    plt.show()


### Token Models ###
token_test = token_df.loc[test_tokens]

# VAN Tokens ONLY
plot_validation(
        token_test.loc[token_test.vanResponse == True, 'response'].values, 
        token_test.loc[token_test.vanResponse == True, 'prob'].values,
        title = "Van Token Flagging",
        lower_threshold = .4,
        upper_threshold = .75)

# SMS Tokens ONLY
plot_validation(
        token_test.loc[token_test.vanResponse == False, 'response'].values, 
        token_test.loc[token_test.vanResponse == False, 'prob'].values,
        title = "SMS Token Flagging")

### Response Models ###
# VAN Name Model
y_pred_van_name = model_van_name.predict_proba(X_van_test)[:, 1]
plot_validation(y_van_names.values[van_test], y_pred_van_name, 
                title = "Valid names provided in VAN?")

# Tripler Model
y_pred_tripler = model_tripler.predict_proba(X_test)[:, 1]
plot_validation(y_tripler.values[test, 1], y_pred_tripler, title = "Is this a Tripler?")

# Name Provided Model
y_pred_name = model_name.predict_proba(X_test)[:, 1]
plot_validation(y_names.values[test], y_pred_name, 
                title = "Did they provide a valid name?")

# Optout Model
y_pred_optout = model_opt.predict_proba(X_test)[:, 1]
plot_validation(y_optout.values[test, 1], y_pred_optout, title = "Opt Out?")

# Wrong Number Model
y_pred_wrongnumber = model_wrongnumber.predict_proba(X_test)[:, 1]
plot_validation(y_wrongnumber.values[test, 1], y_pred_wrongnumber, title = "Wrong Number?")


################################
# Examples
################################

# Do the names match?
labeled['names_match'] = False
for i,row in labeled.loc[~(labeled.namesClean == "")].iterrows():
    labeled.loc[i, 'names_match'] = set(row['names_extract'].split("|")) == set(row['namesClean'].split("|"))

labeled.loc[labeled.probability_name_provided < .5, 'names_extracted'] = ''
labeled.loc[~(labeled.namesClean == "") ,'names_match'].value_counts()

labeled.loc[~(labeled.namesClean == "") & ~(labeled.names_extracted == "") ,'names_match'].value_counts()


labeled.loc[~(labeled.namesClean == "") & 
            (labeled['names_match'] == False), 
            ['voterFinal', 'voterPost','names_extract', 'namesClean', 'probability_name_provided']].to_csv(Path(home, "examples.csv"), index = False)

# View weird cases
labeled['probability_name_provided'] = model_name.predict_proba(X)[:,1]
labeled.loc[test].loc[(y_pred[:, 1] < .6) & 
           (y_names.values[test] == 1), ['voterFinal', 'names', 'probability_name_provided']]