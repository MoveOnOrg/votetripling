#' This file takes in a csv of raw text messages and outputs a 
#' standardized aggregation of conversations. Manipulate variables
#' Anywhere there is a '<<<'
library(data.table)

##############################
# Part I: Aggregate Data
##############################

# 1. Change each default variable to be the correct column name <<<
#' Aggregates conversations into a single row
#' @param messages      the raw data table
#' @param messageIdCol  column name for unique id for each message
#' @param idCol         column name for unique id for each conversation
#' @param dirCol        column name for direction of the message (inbound vs. outbound)
#' @param bodyCol       column name for text message body
#' @param phoneCol      column name for phone number, or whatever column may be needed to join this conversation to another dataset
#' @param inbound       string representing the code for inbound calls
#' @param outbound      string representing the code for outbound calls
aggregateMessages <- function(messages, 
                              messageIdCol = "MessageId",
                              idCol = "ConversationId", 
                              dirCol = "MessageDirection",
                              bodyCol = "MessageBody",
                              phoneCol = "phonenumber",
                              inbound = "inbound",
                              outbound = "outbound") {
  messages[, direction := tolower(eval(as.name(dirCol)))]
  messages[, by = idCol, messageOrder := rank(eval(as.name(messageIdCol)))]
  messages[, by = c(idCol, dirCol), messageOrderDir := rank(eval(as.name(messageIdCol)))]
  messages[, messageOrderDirRev := messageOrder - messageOrderDir]
  aggMessages <- messages[, by = idCol,
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

# 2. Change each default variable to be the correct regex to find scripted answers <<<
#' Tags key elements of the text banker response
#' @param messagesAgg     the data table of aggregated conversations
#' @param negReg          regex of likely negative terms from the text banker (shouldn't change)
#' @param posReg          regex of likely negative terms from the text banker (shouldn't change)
#' @param affirmReg       regex indicating the scripted response of a text banker to a tripler. 
#'                        Use the smallest key phrase possible as there tends to be great variance
#' @param finalAffirmReg  regex indicating the scripted response of a text banker to a tripler after names are supplied. 
#'                        Use the smallest key phrase possible as there tends to be great variance
categorizeBankerResponse <- function(messagesAgg, 
                                     negReg = "remove you|apolog|sorry", 
                                     posReg = "^great|^thank you",
                                     affirmReg = "what are the [a-z]+ names",
                                     finalAffirmReg = "list of candidates|openprogress\\.com|a reminder to vote can make all the difference"
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

# 3. Put the appropriate file path to the raw text data here <<<
data <- fread("./Documents/Personal/Voting/Data/SMS/Test_Data/testdata.csv")

# Aggregate data
agg <- aggregateMessages(data)

# Code Responses
final <- categorizeBankerResponse(agg)

# 4. Put the appropriate output file path here
fwrite("./Documents/Personal/Voting/Data/testdata_aggregated.csv")
