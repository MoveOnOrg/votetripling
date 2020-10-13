###########################################################################################
###########################################################################################
# READ ME
#
#' This file takes in a csv of raw text message transcripts (from, e.g., ThruText, Hustle, Spoke)
#	 and outputs a standardized aggregation of conversations. This aggregation can then be fed to 
#' VoteTripling.org Python cleaning scripts to generate lists of triplers.
#
#  The input file should be a complete transcript of _all_ messages sent during the SMS campaign -
#  those sent by voters as well as textbankers. This is a standard export from all texting platforms.
#  The input file is *not* a set of data items created by volunteers, and does *not* contain only
#  messages sent by triplers.
#
#' Before running this script, you need to do 4 things. All 4 areas are marked
#				with "<<<<<<<<<<" below where action is required.
# 1. LINE 41: Put the filepath for the input file at #1 below. This is a file in civis
#				containing the raw data transcript. Include the schema. If your schema
#				is "ab" then enter as, e.g., "ab.rawsms_1015_wisconsin"
# 2. LINE 44: Put the filepath for the output file at #2 below. This is where in civis the
#				aggregated file will go. Include the schema. Enter as, e.g.,
#				"ab.sms_1015_wisconsin_aggregated"
# 3. LINE 51: Enter the column names from your raw file
#				*in some cases the names as given below will be correct already*
#				please confirm that the column names are correct or the script will not run correctly
# 4. LINE 107: Enter strings that indicate likely responses
#				the python cleaning scripts rely on information contained in texts from voters
#			  as well as texts sent back from textbankers. If the textbanker says "Sorry, I'll remove you"
#				for example, the script will conclude there was an optout. In this section, you can
#				specify phrases your textbankers were likely to have used in various scenarios, based on
#				the recommended replies in your campaign. If, for example, the recommended reply to people
#				who said yes to tripling was "Woohoo!" then you might set "Woohoo!" as a string indicating
#				a likely tripler. You can, in most cases, leave the default values here unchanged, 
#				unless you had especially unique SMS scripts.
###########################################################################################
############################################################################################

library(civis)
install.packages("data.table")
library(data.table)

# 1. Put the input file below: <<<<<<<<<< ACTION NEEDED
inputfile <- ("gz.vvv_smsresults_1005")

# 2. Put the output file below: <<<<<<<<<< ACTION NEEDED
outputfile <- ("gz.vvv_smsresults_1005_agg3")

##############################
# Part I: Aggregate Data
##############################

# 3. Enter correct column names <<<<<<<<<< ACTION NEEDED

# Use this area to ensure that the script can recognize the column names in your export
# Replace the text in quotes below (starting at line 72) with the relevant column names in your raw dataset
#	The names as given will likely be correct for some ThruText exports

# ALL ENTRIES SHOULD BE ALL LOWER CASE, REGARDLESS OF THE ORIGINAL

#' messageIdCol:			 column name for unique id for each *message*
#' idCol:			         column name for unique id for each *conversation*
#' dirCol:		         column name for direction of the message (inbound vs. outbound)
#' bodyCol:			       column name for text message body
#' phoneCol:		       column name for phone number, or any other ID that may be needed
#' inbound:			       string representing the label for inbound SMS
#														that is, in the raw dataset, how are inbound SMS labeled
#														in the 'messagedirection' column? usually inbound/incoming
#' outbound      			 string representing the label for outbound SMS
#														that is, in the raw dataset, how are outbound SMS labeled
#														in the 'messagedirection' column? usually outbound/outgoing

aggregateMessages <- function(messages, 
                              messageIdCol = "messageid",
                              idCol = "conversationid", 
                              dirCol = "messagedirection",
                              bodyCol = "messagebody",
                              phoneCol = "endpointareacode",
                              inbound = "inbound",
                              outbound = "outbound") {
  messages[, direction := tolower(eval(as.name(dirCol)))]
  messages[, by = idCol, messageOrder := rank(eval(as.name(messageIdCol)))]
  messages[, by = c(idCol, dirCol), messageOrderDir := rank(eval(as.name(messageIdCol)))]
  messages[, messageOrderDirRev := messageOrder - messageOrderDir]
  messages[, conversationid := eval(as.name(idCol))]
  aggMessages <- messages[, by = "conversationid",
                          list(
                            contact_phone = max(eval(as.name(phoneCol))),
                            totalMessages = .N,
                            tripleMessage = eval(as.name(bodyCol))[messageOrder == 1 ],
                            voterResponse = paste(eval(as.name(bodyCol))[messageOrderDirRev == 1 & direction == inbound], collapse = "\n"),
                            tripleResponse = max(eval(as.name(bodyCol))[messageOrderDir == 2 & direction == outbound]),
                            voterFinal = paste(eval(as.name(bodyCol))[messageOrderDirRev == 2 & direction == inbound], collapse = "\n"),
                            tripleFinal = max(eval(as.name(bodyCol))[messageOrderDir == 3 & direction == outbound]),
                            voterPost = paste(eval(as.name(bodyCol))[messageOrderDirRev == 3 & direction == inbound], collapse = "\n")
                          )]
  aggMessages <- aggMessages[totalMessages > 1]
  aggMessages
}

##############################
# Part II: Code the responses of the text banker
##############################

#' Cleans a string to make it easier to find phrases
cleanString <- function(string) {
  trimws(tolower(gsub("[^a-zA-Z\\. ]", "", string)))
}

# 4. Change each default variable to be the correct regex to find scripted answers <<<<<<<<<< ACTION NEEDED

# This section is somewhat optional. If you want to add additional search terms, use
# a pipe - | - between each token. Tokens are regex searches.

#' negReg          regex of likely negative terms from the text banker (shouldn't change)
#												these are terms the textbanker is likely to have sent when the voter
#												opted out, was not interested, or was a wrong number
#' posReg          regex of likely positive terms from the text banker (shouldn't change)
#												these are terms the textbanker is likely to have sent when the voter
#												agreed to be a tripler
#' affirmReg       regex indicating the scripted response of a text banker to a tripler. 
#'                      Use the smallest key phrase possible as there tends to be great variance
#'											If your response to a 'yes' was 'woohoo!' then add "|woohoo!" to affirmReg terms
#' finalAffirmReg  regex indicating the scripted response of a text banker to a tripler after names are supplied.
#'											That is, specify here a portion of the response that was sent to triplers after
#'											providing names. If your final signoff was "a reminder to vote can make all the difference
#'											coming from a friend" then you might put a portion of that in finalAffirmReg.
#'                      Use the smallest key phrase possible as there tends to be great variance
categorizeBankerResponse <- function(messagesAgg, 
                                     negReg = "remove you|apolog|sorry", 
                                     posReg = "^great|^thank you",
                                     affirmReg = "what are the [a-z]+ names",
                                     finalAffirmReg = "a reminder to vote can make all the difference"
) {
  
  # No response
  messagesAgg[, noResponse := (totalMessages <= 2)]
  
  # Clean all responses
  # messagesAgg[, tripleResponseClean := cleanString(tripleResponse)]
  # messagesAgg[, tripleFinalClean := cleanString(tripleFinal)]
  # messagesAgg[, tripleMessageClean := cleanString(tripleMessage)]
  # messagesAgg[, voterResponseClean := cleanString(voterResponse)]
  # messagesAgg[, voterFinalClean := cleanString(voterFinal)]
  # messagesAgg[, voterPostClean := cleanString(voterPost)]
  
  # Response with removal promise
  messagesAgg[, negResponse := grepl(negReg, cleanString(tripleResponse)) | grepl(negReg, cleanString(tripleFinal))]
  
  # Response with affirmation
  messagesAgg[, posResponse := grepl(posReg, cleanString(tripleResponse)) | grepl(posReg, cleanString(tripleFinal))]
  messagesAgg[, affirmResponse := grepl(affirmReg, cleanString(tripleResponse)) | grepl(affirmReg, cleanString(tripleFinal))]
  messagesAgg[, finalAffirmResponse := grepl(finalAffirmReg, cleanString(tripleResponse)) | grepl(finalAffirmReg, cleanString(tripleFinal))]
  
  messagesAgg
}

##############################
# Part III: Run Aggregation and Write File
##############################

# Read in data
data2 <- read_civis(paste(inputfile))
data <- data.table(data2)

# Aggregate data
agg <- aggregateMessages(data)

# Code Responses
final <- categorizeBankerResponse(agg)

# Write data
write_civis(final,paste(outputfile))
