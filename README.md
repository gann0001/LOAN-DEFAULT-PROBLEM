# LOAN-DEFAULT-PROBLEM

Kaggle Competition: https://www.kaggle.com/c/ou-loan-default-problem

File descriptions:

LoanDefault-Train.csv - the training set: 16,000 records

LoanDefault-Test.csv - the test set: 14,000 records

submission_file.csv - a sample submission file in the correct format -- make sure you upload probabilities!

Data fields:
Limit : total credit limit for the individual

Gender : M=male, F= Female

Education : GradS = graduate degree, Univ = undergraduate degree, HS = highschool diploma, Others = other

MarriageStatus : married, single, other

Age : age of individual

Status1 - Status6 : repayment status 1 - 6 months prior (-2 = no payment needed; -1 = paid in full; 0 = revolving credit; 1 = 1 month delay in payment, 2 = 2 month dealy in payment, etc.)

Bill1 - Bill6 : bill statement amount for 1 - 6 months prior

Payment1 - Payment6 : payment amount for 1 - 6 months prior

IndLevel : external data associated with individual

Default : Whether or not the individual defaults on the loan this month (Y=yes, N = no) -- target for your classification problem
