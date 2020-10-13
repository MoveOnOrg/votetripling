# Votetripling Extraction Script Instructions
This document describes how to use 5 different scripts for cleaning/aggregating vote tripling SMS data. Please find your use case below and follow the instructions.  
  
## SMS Aggregation
**Use Case:** I need to aggregate SMS messages by conversation. This step is necessary before performing any extraction on SMS data.  
  
**Inputs:**
Add a dataset to civis. This data should consist of raw individual SMS messages, not grouped by conversation. The columns needed will be specified within the R script below.

**Instructions:**
Open the script aggregate_text_messages.R and follow the instructions in that script to aggregate messages into a single row per conversation

**Outputs:**
A file (filename specified by you in the R script) with a single row representing each text message conversation, including the following fields
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
First follow the instructions above in the **SMS Aggregation** section. The output of that step will provide a dataset which will be used as input here.

**Instructions:**
In the _VoteTripling.org Pledge Cleaning Scripts_ project, run the container script titled _2. Pledges from SMS Transcripts_. Provide the name of your input dataset and the names of your output datasets (including schema names) as parameters.

**Outputs:**
This script will output two datasets:  
1. A file of triplers called `sms_triplers`. For each tripler, we provide the following fields (each row represents one text message conversation):
- *ConversationId* a unique identifier for the conversation
- *contact_phone* the phone number of the target 
- *is_tripler* did this person agree to be a tripler ('yes' for everyone in this file)
- *opted_out* did this person opt out of future messages
- *wrong_number* did we have the wrong number for this person
- *names_extract* what names (if any) were provided by this person as tripling targets

2. A file of conversations for manual review called `sms_manual_review`, with the following fields:
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
Add a dataset to civis containing the column 'names' containing the names of tripler targets logged by a text banker from SMS conversations.

**Instructions:**
In the _VoteTripling.org Pledge Cleaning Scripts_ project, run the container script titled _3. Pledges from Generic Volunteer Data Entry._. Provide the name of your input dataset and the names of your output dataset (including schema names) as parameters.

**Outputs:**
A dataset named `labeled_names_cleaned_no_response` with the cleaned names in a column titles "clean_names", along with any other columns in the initial file 
  
  
## Text Banker Log Cleaning (utilizing text message conversation)
**Use Case:** I have text banker logs for names provided by vote triplers. I also have access to the initial text conversation. I need these logs cleaned up and standardized. We use a different script for these cases, because we can clean up the logs better and perform spell check by looking at the original messages.  
  
**Inputs:**
First follow the instructions above in the **SMS Aggregation** section. The output of that step will provide a dataset which will be used as input here.  
Next join text banker logs to each conversation by your conversation id. Preserve all of the columns in the aggregated dataset and make sure that the text banker logs are in a column titled 'names'.

**Instructions:**
In the _VoteTripling.org Pledge Cleaning Scripts_ project, run the container script titled _4. Pledges from Generic Volunteer Data Entry and SMS Transcript_. Provide the name of your input dataset and the names of your output dataset (including schema names) as parameters.

**Outputs:**
A dataset named `labeled_names_cleaned_with_response` with the cleaned names in a column titles "clean_names", along with any other columns in the initial file
  
  
## VAN Export Cleaning
**Use Case:** I have a VAN Export and I need to extract any tripling target names from the note text.

**Inputs:**
Add a dataset to civis containing the following columns:
- *VANID* a unique ID for this row
- *ContactName* the name of the tripler
- *NoteText* free text possibly including names of tripling targets

**Instructions:**
In the _VoteTripling.org Pledge Cleaning Scripts_ project, run the container script titled _5. Pledges from VAN Comments_. Provide the name of your input dataset and the names of your output dataset (including schema names) as parameters.

**Outputs:**
This script will output two datasets:  
1. A file of triplers called `van_cleaned`. For each tripler, we provide the following fields (each row represents one text message conversation):
- *VANID* a unique identifier for the conversation
- *names_extract* the extracted names

2. A file of conversations for manual review called `van_manual_review`, with the following fields:
- *VANID* a unique identifier for the conversation
- *ContactName* a unique identifier for the conversation
- *NoteText* free text possibly including names of tripling targets
- *names_extract* a guess for the extracted names (to be reviewed)
