# Votetripling Extraction Script Instructions
This document describes how to use 5 versions of name extraction scripts for vote tripling SMS data. Please find your use case below and follow the instructions.  
All of these scripts should be run out of the directory `Projects/NLP/SMS_Annotation`. 
All input data should be added to `Projects/NLP/SMS_Annotation/Input_Data`. 
All output data (after running a script) will be found in `Projects/NLP/SMS_Annotation/Output_Data`. 
  
## SMS Aggregation
**Use Case:** I need to aggregate SMS messages by conversation. This step is necessary before performing any extraction on SMS data.  
  
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
  
  
## SMS Conversation Categorization and Name Extraction
**Use Case:** I have SMS conversations and I need to figure out which text recipiants volunteered to triple, which chose to opt out, what names they provided, and whether they moved.  
  
**Inputs:**
Add a csv to the Input_Data folder. This csv file must be of the same format as the output of the aggregation in step 1.   

**Instructions:**
In this directory, run `python3 Code/annotate_conversations.py -d [input_filename]`. 

**Outputs:**
This script will output two files:  
1. A file of triplers. For each tripler, we provide the following fields (each row represents one text message conversation):
- *ConversationId* a unique identifier for the conversation
- *contact_phone* the phone number of the target 
- *is_tripler* did this person agree to be a tripler ('yes' for everyone in this file)
- *opted_out* did this person opt out of future messages
- *wrong_number* did we have the wrong number for this person
- *names_extract* what names (if any) were provided by this person as tripling targets

2. A file of conversations for manual review, with the following fields:
- *ConversationId* a unique identifier for the conversation
- *contact_phone* the phone number of the target 
- *voterResponse* initial response(s) by the target (generally where the target makes known if they opt out or want to triple)
- *voterFinal* the final follow up message sent by the target (generally where they provide names)
- *voterPost* post script from the target (generally a thank you or good luck)
- *is_tripler* guess for did this person agree to be a tripler (to be reviewed)
- *opted_out* guess for did this person opt out of future messages (to be reviewed)
- *wrong_number* guess for did we have the wrong number for this person (to be reviewed)
- *names_extract* guess for what names (if any) were provided by this person as tripling targets (to be reviewed)

  
## Text Banker Log Cleaning
**Use Case:** I have text banker logs for names provided by vote triplers. I need these logs cleaned up and standardized.  
  
**Inputs:**
Add a csv to the Input_Data folder. This csv file must contain column 'names' containing the names logged by a text banker  

**Instructions:**
In this directory, run `python3 Code/name_cleaning.py -d [input_filename]`  

**Outputs:**
A File with the cleaned names in a column titles "clean_names", along with any other columns in the initial file 
  
  
## Text Banker Log Cleaning (utilizing text message conversation)
**Use Case:** I have text banker logs for names provided by vote triplers. I also have access to the initial text conversation. I need these logs cleaned up and standardized. We use a different script for these cases, because we can clean up the logs better and perform spell check by looking at the original messages.  
  
**Inputs:**
Add a csv to the Input_Data folder. 
This csv file must be of the same format as the output of the aggregation in step 1.
This csv file must also contain column 'names' containing the names logged by a text banker.

**Instructions:**
In this directory, run `python3 Code/name_cleaning_with_responses.py -d [input_filename]`

**Outputs:**
A File with the cleaned names in a column titles "clean_names", along with any other columns in the initial file
  
  
## VAN Export Cleaning
**Use Case:** I have a VAN Export and I need to extract any tripling target names from the note text.

**Inputs:**
Add a csv to the Input_Data folder. This csv file must contain the following columns:
- *VANID* a unique ID for this row
- *ContactName* the name of the tripler
- *NoteText* free text possibly including names of tripling targets

**Instructions:**
In this directory, run `python3 Code/van_export_cleaning.py -d [input_filename]`  

**Outputs:**
This script will output two files:  
1. A file of triplers. For each tripler, we provide the following fields (each row represents one text message conversation):
- *VANID* a unique identifier for the conversation
- *names_extract* the extracted names

2. A file of conversations for manual review, with the following fields:
- *VANID* a unique identifier for the conversation
- *ContactName* a unique identifier for the conversation
- *NoteText* free text possibly including names of tripling targets
- *names_extract* a guess for the extracted names (to be reviewed)
