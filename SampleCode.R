#################################
#######  SAMPLE CODE   ##########
#################################
library(data.table)
# Open CSV file and convert to data table
# Full path to your csv file
fullPath = <The path to the provided csv file goes here>
DataSet = data.table(read.csv(file = fullPath))
# We have a panel data.table with information about dates, Entity IDs, Returns (T1 and T0), and 36 different indicators).


# Example on how to check the performance of a given prediction.
# Lets say we are using the sign of an indicator(GROUP_E_ALL_SG90) as predictor
# We select only the data we are interested in
PredictionData = subset(DataSet, select = c('DATE', 'RP_ENTITY_ID','GROUP_E_ALL_SG90', 'T1_RETURN'))
# We drop NA's
PredictionData = PredictionData[complete.cases(PredictionData)]
# We get unique rows
PredictionData = unique(PredictionData)
# We sort by DATE, setting its key
setkey(PredictionData,'DATE')
# Compute Average Return per day using the sign of the indicator as predictor
PredictionData[, AVGRET:= mean(sign(GROUP_E_ALL_SG90)*T1_RETURN,na.rm = TRUE), by = c('DATE')]
Results = subset(PredictionData,select = c('DATE','AVGRET'))
Results = unique(Results)
# We plot the Cummulative Log Returns
plot( as.Date(Results$DATE), cumsum(log(Results$AVGRET+1)), t = 'l', col = 'blue', ylab = 'Cummulative Return', xlab = 'DATE')
# Some Stats
# AnnualizedReturn
AnnualizedReturn = mean(log(Results$AVGRET+1))*252 
# AnnualizedVolatility 
AnnualizedVolatility = sqrt(var(log(Results$AVGRET+1)))*sqrt(252)
#↓ Information Ratio     
InformationRatio = AnnualizedReturn/AnnualizedVolatility
title(paste('Return Profile - Information Ratio', round(InformationRatio,2)))
