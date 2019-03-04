########## Association Rules for Primate Results #######
rm(list=ls())

library(arules)
results<-read.csv("~/2016_Presidential_Elections/RepublicanPrimaryAnalysis/AssociationRules/Cruz Categorical Final.csv")

#let us take a look at the data and figure out if there are any variables we need to convert to factors

head(results)
str(results)

#convert Outcome, HigherEducation, PerCapitaIncome, PopulationDensity, BelowPovertyLine, WhiteOnly into factors and verify
results$HigherEducation <- as.factor(results$HigherEducation)
results$PerCapitaIncome <- as.factor(results$PerCapitaIncome)
results$PopulationDensity <- as.factor(results$PopulationDensity)
results$BelowPovertyLine <- as.factor(results$BelowPovertyLine)
results$WhiteOnly <- as.factor(results$WhiteOnly)

summary(results)
str(results)

#winning outcome dimensions
dim(results[results$Outcome=="Won",])
dim(results)

# In order to run the association rules algorithm, the data frame of factors needs to be converted to 
# a list of transactions

resultstrans<-as(results,"transactions")

# The following line actually runs the algorithm to generate the rules
# minlen is the minimum "length" of a rule allowed
# A single antecedent implies a single consequent results in a length of 2
#run apriori algorithm- At 0.05 support->conf of 0.15 was selected after trial/error. a conf value>0.19 will not yield win rules and a value
#less than 0.15 results in unnecessary rules with a lift < 1, which does not add much value

electionrules<-apriori(resultstrans, parameter=list(supp=0.1, conf=0.3, minlen=2))

# We can get a summary, including the number of rules generated, with the following statement
summary(electionrules)

# We can sort the rules by a desired metric - here we sort them by lift
ruleslift<-sort(electionrules, decreasing=TRUE, by="lift")

# We can also store the rules in a data frame, which we're a little more familiar with (manipulating, sorting, etc.)
rulesliftdf<-as(ruleslift, "data.frame")

# We're particularly interested in rules that give us information about when the outcome of the Rexnord transaction
# is positive for the company(Won). We can isolate these particular rules using the following statements
electionrulesWon <- subset(electionrules, subset = rhs %pin% "Outcome=Won")

# Check how many rules we've generated
summary(electionrulesWon)

# Sort from highest lift to lowest
electionrulesWonlift<-sort(electionrulesWon, decreasing=TRUE, by="lift")

# Look at the rules, their support, confidence and lift
#inspect(electionrulesWonlift)

# Let's convert this to a dataframe so we can manipulate it more easily
electionrulesWonliftdf<-as(electionrulesWonlift, "data.frame")

# Look at the first 10 rules
print(electionrulesWonliftdf[1:10,])


# Now isolate the rules where rexnord quote resulted in a losing outcome
electionrulesLost <- subset(electionrules, subset = rhs %pin% "Outcome=Lost")

# Look at a summary
summary(electionrulesLost)

# Sort by lift
electionrulesLostlift<-sort(electionrulesLost, decreasing=TRUE, by="lift")

# Let's convert this to a dataframe so we can manipulate it more easily
electionrulesLostliftdf<-as(electionrulesLostlift, "data.frame")

# Look at the first 10 rules
print(electionrulesLostliftdf[1:10,])

write.csv(electionrulesLostliftdf[1:10,],file = "~/2016_Presidential_Elections/RepublicanPrimaryAnalysis/AssociationRules/Generated Rules/Kasich Loss Rules.csv")
write.csv(electionrulesWonliftdf[1:10,],file = "~/2016_Presidential_Elections/RepublicanPrimaryAnalysis/AssociationRules/Generated Rules/Kasich Won Rules.csv")
