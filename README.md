# votetripling
This document describes how to use 4 versions of name extraction scripts for vote tripling SMS data. Please find your use case below and follow the instructions.

1. I need to aggregate SMS messages by conversation. This step is necessary before performing any extraction on SMS data.  
**Inputs:**
Add a csv  to the Input_Data folder. This csv should be raw individual SMS messages, not grouped by conversation.  

**Instructions:**
Open the script aggregate_text_messages.R in RStudioo and follow the instructions to aggregate messages into a single row per conversation

**Outputs:**
A file with a single row representing each text message conversation, including the following fields
- *ConversationId* a unique identifier for the conversation
- *contact_phone* the phone number of the target 
- *totalMessages* the total number of messages exchanged
- *tripleMessage* initial message sent from the text banker to the target
- *voterResponse* initial response(s) by the target (generally where the target makes known if they opt out or want to triple)
- *tripleResponse* follow up message sent from the text banker to the target
- *voterFinal* the final follow up message sent by the target (generally where they provide names)
- *tripleFinal* final follow up sent by text banker
- *voterPost* post script from the target (generally a thank you or good luck)
- *noResponse* boolean for whether there was no response
- *negResponse* boolean for generally negative or discouraging terms (sorry, no, etc.)
- *posResponse* boolean for generally positive or encouraging terms
- *affirmResponse* boolean for presence of a scripted affirmation by text banker
- *finalAffirmResponse* boolean for presence of a scripted follow up affirmation by text banker

2. I have SMS conversations and I need to figure out which text recipiants volunteered to triple, which chose to opt out, what names they provided, and whether they moved.  
**Inputs:**
Add a csv to the Input_Data folder. This csv file must be of the same format as the output of the aggregation in step 1.   

**Instructions:**
In this directory, run `python3 annotate_conversations.py -d [input_filename]`. 

**Outputs:**
This script will output two files:
a. A file of triplers. For each tripler, we provide the following fields:
A file with a single row representing each text message conversation, including the following fields
- *ConversationId* a unique identifier for the conversation
- *contact_phone* the phone number of the target 
- *is_tripler* did this person agree to be a tripler ('yes' for everyone in this file)
- *opted_out* did this person opt out of future messages
- *wrong_number* did we have the wrong number for this person
- *names_extract* what names (if any) were provided by this person as tripling targets

b. A file of conversations for manual review, with the following fields:
- *ConversationId* a unique identifier for the conversation
- *contact_phone* the phone number of the target 
- *voterResponse* initial response(s) by the target (generally where the target makes known if they opt out or want to triple)
- *voterFinal* the final follow up message sent by the target (generally where they provide names)
- *voterPost* post script from the target (generally a thank you or good luck)
- *is_tripler* guess for did this person agree to be a tripler
- *opted_out* guess for did this person opt out of future messages
- *wrong_number* guess for did we have the wrong number for this person
- *names_extract* guess for what names (if any) were provided by this person as tripling targets


3. I have text banker logs for names provided by vote triplers. I need these logs cleaned up and standardized.  
**Inputs:**
Add a csv to the Input_Data folder. This csv file must contain column 'names' containing the names logged by a text banker  

**Instructions:**
In this directory, run `python3 name_cleaning.py -d [input_filename]`  

**Outputs:**
A File with the cleaned names in a column titles "clean_names", along with any other columns in the initial file 

4. I have text banker logs for names provided by vote triplers. I also have access to the initial text conversation. I need these logs cleaned up and standardized. We use a different script for these cases, because we can clean up the logs better and perform spell check by looking at the original messages.  
**Inputs:**
Add a csv to the Input_Data folder. 
This csv file must be of the same format as the output of the aggregation in step 1.
This csv file must also contain column 'names' containing the names logged by a text banker.

**Instructions:**
In this directory, run `python3 name_cleaning_with_responses.py -d [input_filename]`

**Outputs:**
A File with the cleaned names in a column titles "clean_names", along with any other columns in the initial file
