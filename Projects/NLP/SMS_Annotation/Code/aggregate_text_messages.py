#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:37:08 2021

    This file takes in a csv of raw text messages and outputs a 
    standardized aggregation of conversations

@author: alutes
"""

import re
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path
import tqdm
import argparse

def clean_string(string):
    """ Cleans a string to make it easier to find phrases """
    if string:
        return re.sub("[^a-z\\. ]", "", string.lower().strip())
    return ''

def apply_parallel(dfGrouped, func, max_cpu = 8):
    num_cpu = min(max_cpu, multiprocessing.cpu_count())
    retLst = Parallel(n_jobs=num_cpu)(delayed(func)(group) for name, group in dfGrouped)
    return retLst

def head(array):
    if len(array) > 0:
        return array[0]
    return 

def aggregate_conversation(
        conversation,
        initial_triple_phrase,
        message_id_col,
        id_col,
        dir_col,
        body_col,
        phone_col,
        inbound,
        outbound
        ): 
    """ Aggregates a single conversation """
    
    # Eliminate messages before the first triple message
    if initial_triple_phrase:
        first_messages = conversation.loc[(~(conversation[body_col].isnull()) &
                                       conversation[body_col].str.contains(initial_triple_phrase))]
        first_id = first_messages[message_id_col].min()
        conversation = conversation.loc[conversation[message_id_col] >= first_id]

    # Don't bother for an empty conversation (or one without an opening message)
    if len(conversation) < 2:
        return
    
    # Evaluate message order
    conversation['messageOrder']  = conversation[message_id_col].rank()
    
    # Find message order within each direction
    conversation.loc[conversation[dir_col] == inbound, 'messageOrderDir']  = conversation.loc[conversation[dir_col] == inbound, message_id_col].rank()
    conversation.loc[conversation[dir_col] == outbound, 'messageOrderDir']  = conversation.loc[conversation[dir_col] == outbound, message_id_col].rank()
    conversation['messageOrderDirRev'] = conversation['messageOrder'] - conversation['messageOrderDir']
    
    # Add other information
    message_bodies = conversation[body_col].values
    agg = {
            'contactPhone' : max(conversation[phone_col]),
            'totalMessages' : len(conversation),
            'tripleMessage' : head(message_bodies[conversation['messageOrder'] == 1]),
            'voterResponse' : '\n'.join(message_bodies[(conversation['messageOrderDirRev'] == 1) & (conversation[dir_col] == inbound)]),
            'tripleResponse' : head(message_bodies[(conversation['messageOrderDir'] == 2) & (conversation[dir_col] == outbound)]),
            'voterFinal' : '\n'.join(message_bodies[(conversation['messageOrderDirRev'] == 2) & (conversation[dir_col] == inbound)]),
            'tripleFinal' : head(message_bodies[(conversation['messageOrderDir'] == 3) & (conversation[dir_col] == outbound)]),
            'voterPost' : '\n'.join(message_bodies[(conversation['messageOrderDirRev'] == 3) & (conversation[dir_col] == inbound)])
            }
    return agg    
    
def aggregate_messages(
        messages,
        initial_triple_phrase,
        message_id_col,
        id_col,
        dir_col,
        body_col,
        phone_col,
        inbound,
        outbound
        ): 
    """ Aggregates each sms conversation into a single row
        
        
        Attributes
        ----------
        messages:
            pandas dataframe with raw data for sms messages. Expects one message per row
        initial_triple_phrase : 
            regex used to identify the initial message from a text banker
        message_id_col :
            column name for unique id for each message
        id_col :
            column name for unique id for each conversation
        dir_col : 
            column name for direction of the message (inbound vs. outbound)
        body_col : 
            column name for text message body
        phone_col : 
            column name for phone number, or whatever column may be needed to join this conversation to another dataset
        inbound : 
            string representing the code for inbound calls
        outbound : 
            string representing the code for outbound calls
            
            
        Returns a pandas DataFrame with the columns listed below. 
        Each row tracks a conversation between a 'phone banker' who is a volunteer asking voters to volunteer to triple
        and a 'voter' from our contact list.
        ----------
        conversationId:
            unique id for the conversation
        contactPhone: 
            phone number of the voter
        totalMessages:
            number of messages in the conversation
        tripleMessage: 
            initial message sent by the phone banker to the voter 
        voterResponse:
            all responses by the voter between the initial phonk banker message and their first follow up
        tripleResponse: 
            first follow up by the phone banker
        voterFinal:
            all responses by the voter between the second phonk banker message and their final follow up
        tripleFinal : 
            third and (usually) final follow up by the phone banker
        voterPost :
            concatenation of all messages sent by the voter after tripleFinal
    """

    # Clean up columns and rename a few of them
    messages.loc[messages[body_col].isnull(), body_col] = ''
    messages['direction'] = messages[dir_col].str.lower()
    messages['conversationId'] = messages[id_col]
    messages.index = messages[id_col]
    messages = messages.sort_index()
    
    # Create temp function to incorporate all arguments but the data
    def agg_conversation(
        conversation
        ):
        try:
            agg = aggregate_conversation(
                conversation,
                initial_triple_phrase,
                message_id_col,
                id_col,
                dir_col,
                body_col,
                phone_col,
                inbound,
                outbound
                )
            return agg
        except TypeError as e:
            print("FAILURE AT CONVERSATION {i}".format(i=row.name))
            print(e)
            print(conversation)
            print("\n")

    # @note: The reason this looks grotesque is because it is
    # pandas.groupby was wayyyy too slow even when I manually added parallelism
    # this step took about 30s in R but still takes ~20m locally
    # but hey that's better than 2 hours
    all_agg = []
    i_last = messages.index[0]
    conversation = []
    for i, row in tqdm.tqdm(messages.iterrows(), total = len(messages)):
        if i == i_last:
            conversation.append(row.to_dict())
        elif conversation: 
            agg_conv = agg_conversation(pd.DataFrame(conversation))
            if agg_conv:
                all_agg.append(agg_conv)
            elif len(conversation) > 1:
                break
            conversation = [row.to_dict()]
        i_last = i

    agg_df = pd.DataFrame(all_agg)
    return agg_df
    
def categorize_banker_response(
     agg_df,
     affirm_regex,
     affirm_regex_final,
     neg_reg = "remove you|apolog|sorry",
     pos_reg = "^great|^thank you"
     ):
    """ Tags key elements of the text banker response
    
        Attributes
        ----------
        negReg:
            regex used to identify very very common patterns for negative or apologetic messages not in the script
        posReg : 
            regex used to identify very very common patterns for encouraging messages not in the script
            (we don't bother with a full nlp model for this since the text bankers are more consistent)
        affirmReg :
            described in help docs for the args
        finalAffirmReg :
            described in help docs for the args
            
            
        Returns a pandas DataFrame with the columns listed above in aggregate_messages plus the additions below: 
        ----------
        noResponse:
            no response by the text banker
        negResponse: 
            text banker gave negative message likely to indicate a non-tripler
        posResponse:
            text banker gave encouraging message likely to indicate a tripler
        affirmResponse: 
            text banker gave something similar to the scripted response for a tripler
        finalAffirmResponse:
            text banker gave something similar to the scripted final response for a tripler
    """
    
    # No response
    agg_df['noResponse'] = agg_df['totalMessages'] <= 2
    
    # Response with removal promise
    agg_df['negResponse'] = agg_df.apply(
          lambda x: clean_string(x['tripleResponse']) + ' ' + clean_string(x['tripleFinal']),
          axis = 1
          ).str.contains(neg_reg)

    # Response with affirmation
    agg_df['posResponse'] = agg_df.apply(
          lambda x: clean_string(x['tripleResponse']) + ' ' + clean_string(x['tripleFinal']),
          axis = 1
          ).str.contains(pos_reg)

    agg_df['affirmResponse'] = agg_df.apply(
          lambda x: clean_string(x['tripleResponse']) + ' ' + clean_string(x['tripleFinal']),
          axis = 1
          ).str.contains(affirm_regex)

    agg_df['finalAffirmResponse'] = agg_df.apply(
          lambda x: clean_string(x['tripleResponse']) + ' ' + clean_string(x['tripleFinal']),
          axis = 1
          ).str.contains(affirm_regex_final)
  
    return agg_df

def run_main(
        input_path,
        output_path,
        message_id_col,
        id_col,
        dir_col,
        body_col,
        phone_col,
        inbound,
        outbound,
        affirm_regex,
        affirm_regex_final,
        initial_triple_phrase,
        n_common_responses = 20
        ):
    """ Reads in data from the specified file, performs aggregation and cleaning, 
        and writes an output file """

    # Read in input data
    input_data = pd.read_csv(input_path)
    
    # Check input columns
    for col in [message_id_col,
        id_col,
        dir_col,
        body_col,
        phone_col]:
        if not col in input_data.columns:
            raise ValueError("{col} is not found in the dataset, please examine data or specify a different column name to use".format(col=col))

    # Make sure directions are correct
    for direction in input_data[dir_col].unique():
        if not direction in [inbound, outbound]:
            raise ValueError("{direction} is listed as a message direction. Please specify this as the inbound or outbound type or scrub from data".format(direction=direction))
    
    # Aggregate
    aggregate_data = aggregate_messages(
            input_data,
            initial_triple_phrase,
            message_id_col,
            id_col,
            dir_col,
            body_col,
            phone_col,
            inbound,
            outbound
            )

    # Print the most common responses in case the user wants to change response regexes
    if n_common_responses:
        
        # Common Initial Responses
        print(
            """Below are the {n} most common responses from the text banker to triplers. \n
               It may be useful to add commonly seen phrases to affirmation_regex
           """.format(n=n_common_responses))
        print(aggregate_data["tripleResponse"].value_counts()[0:n_common_responses])
        
        # Common Final Responses
        print(
            """Below are the {n} most common secondary follow up responses from the text banker to triplers. \n
               It may be useful to add commonly seen phrases to affirmation_regex_final
           """.format(n=n_common_responses))
        print(aggregate_data["tripleFinal"].value_counts()[0:n_common_responses])

    # Categorize Responses
    output_data = categorize_banker_response(
            aggregate_data,
            affirm_regex,
            affirm_regex_final
            )

    # Write output data
    output_data.to_csv(output_path, index = False)

def main(args):
    run_main(
            input_path = Path(args.input_data_path),
            output_path = Path(args.output_path),
            message_id_col = args.messageIdCol,
            id_col = args.idCol,
            dir_col = args.dirCol,
            body_col = args.bodyCol,
            phone_col = args.phoneCol,
            inbound = args.inbound,
            outbound = args.outbound,
            affirm_regex = args.affirmation_regex,
            affirm_regex_final = args.affirmation_regex_final,
            initial_triple_phrase = args.initial_triple_phrase
            )

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument(
        "-d", "--input_data_path",
        default="/Users/alutes/Documents/GitHub/votetripling/Projects/NLP/SMS_Annotation/Input_Data/test_input.csv",
        help="file path containing input data"
    )
    PARSER.add_argument(
        "-o", "--output_path",
        default="/Users/alutes/Documents/GitHub/votetripling/Projects/NLP/SMS_Annotation/Input_Data/test_aggregated.csv",
        help="path for where to output aggregated data"
    )
    PARSER.add_argument(
        "-m", "--messageIdCol",
        default="MessageId",
        help="name of the column in input data containing unique Id for each message. This column is used to determine correct order of messages. Timestamp should work fine here."
    )
    PARSER.add_argument(
        "-id", "--idCol",
        default="ConversationId",
        help="name of the column in input data containing unique Id for each conversation"
    )
    PARSER.add_argument(
        "-dir", "--dirCol",
        default="MessageDirection",
        help="name of the column in input data containing direction (inbound or outbound) of each message"
    )
    PARSER.add_argument(
        "-b", "--bodyCol",
        default="MessageBody",
        help="name of the column in input data containing the message body"
    )
    PARSER.add_argument(
        "-p", "--phoneCol",
        default="EndpointPhoneNumber",
        help="name of the column in input data containing the phone number. Any unique identifier for the recipient will suffice"
    )
    PARSER.add_argument(
        "-in", "--inbound",
        default="Inbound",
        help="exact text of the label given to inbound message in the input data"
    )
    PARSER.add_argument(
        "-out", "--outbound",
        default="Outbound",
        help="exact text of the label given to outbound message in the input data"
    )
    PARSER.add_argument(
        "-a", "--affirmation_regex",
        default="what are the [a-z]+ names",
        help="regex used to mark the phrases used by text bankers to affirm that the voter has responded that they are willing to participate"
    )
    PARSER.add_argument(
        "-af", "--affirmation_regex_final",
        default="ways to vote|how to vote|a reminder to vote can make all the difference",
        help="regex used to mark the phrases used by text bankers after to affirm that the voter has followed up to provided names"
    )
    PARSER.add_argument(
        "-t", "--initial_triple_phrase",
        default=None,
        help="regex used to mark the initial intro phrase used by text bankers. We will eliminate any messages before this phrase as anomalous. Leave blank if you don't want to remove anything."
    )
    main(PARSER.parse_args())