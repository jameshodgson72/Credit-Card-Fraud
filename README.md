# Credit Card Fraud Detection

## Motivation
This project was completed as part of the General Assembly Data Science Immersive boot camp training course.  The topic of Credit Card Fraud Detection was selected because of a personal interest in fraud detection, particularly in the context of processing high-volume consumer transactions.

## Repository files
This repository contains 4 notebooks:-
- "Data merging and cleaning":  code for loading data from 3 files, cleaning and merging, ready for use through the rest of the project.
- "EDA - Credit Card Fraud Prediction":  Exploratory data analysis.
- "Models based on standard data":  Fraud predictor models based on the features contained in the data.
- "Models baded on engineered features":  Fraud predictor models based on features contained in the data plus features added in this notebook.

The repository also contains a Python code file, "jh_eda_classes.py" which includes a few standard EDA functions I have built into a class to save time by re-using them from project to project.

The data files are not stored here due to their size, but the data set is available on Kaggle, here -> https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions

## Technology and tools used

- Pandas
- NumPy
- Scikit-learn
- Imbalanced Learn
- Matplotlib
- Seaborn
- SQLite
- XGBoost

## Background

Credit card fraud is when your credit card is stolen and used to make fraudulent purchases.  Increasingly such fraud takes place electronically, i.e. your card details or electronic "signature" are mined and then used without the card being physically stolen or presented at stores.  The fraudster can operate entirely remotely and use your card to pay for services or have goods delivered to pick-up points like Amazon lockers.

Credit card fraud impacts on consumers and credit-card companies.  For consumers it can mean your card it stopped so you can't use it until its replaced.  For credit card companies, they may have to pay the cost of the fraudulent transactions and/or pay the cost of extra measures or back office processes needed to mitigate fraud.

## Project aims

The aim of this project was to build a predictor for credit card fraud, utilising the details of an individual transaction, along with the card and card-holder's specifics.  The predictor should therefore predict whether an individual transaction is fradulent (or not).

The secondary aim of the project was to use Exploratory Data Analysis to gain insights into the kinds of transactions, credit cards, or users which are more prone to fraud.

## Success measures

Primarily I wanted to be able to assess the effectiveness of my classifier.  I therefore used typical classifier metrics, such as precision and recall, to compare results.  AUROC (area-under the receiver-operatoring characteristic graph) was also a useful measure and one which some models could be trained to maximise.

Additionally I wanted to look at the performance of the classifier from a business-suitability perspective, i.e.:-
- what proportion of the fraud cases in the data set did the classifier detect? (This is actually equivalent to recall).
- what proportion of non-fraud cases did the classifier incorrectly flag as fraud?

The former relates to the cost of undetected fraud, while the latter relates to the cost of any unnecessary intervention on non-fraudulent transactions. 

## Data set

The data set used contains over 24 million card transactions relating to 2000 US-based card users and spanning a period of 25 years (1995-2020). It is simulated data, but has been created (by IBM) for the purpose of building machine-learning algorithms to detect fraud. It contains 3 files:-

- Transactions: id of the customer and card, information about the vendor, the date, time and value of the transaction, etc.
- Users: Credit-card company customer data, including demographic information, address, credit scoring, etc.
- Cards: Credit-card details including type and brand of card, security features, etc. The data can be "joined" together using each user's unique id and the credit card numbers.

The transactions contain almost 30,000 fraud cases (just over 0.1%).  As such there is therefore a massive class imbalance between fraud and non-fraud cases, and this will need to be addressed when training machine-learning models.

## Data preparation

A subset (approximately one sixth) of the transaction data was used for this project due to the limited hardware available.  The data sub-set contained 5000 fraud transactions out of around 4.5M.  The complete data for 373 out of the 2000 card users was taken to achieve this.  So a complete transaction history for those 373 users was retained, maintaining the validity of any aggregating, relative or windowing functions applied to the data later on (see engineered features).

Data cleanining work included:
- Filling null values in Zip codes and US states for online and overseas transactions.
- Filling null values in an "error code" field which actually indicated there was no error.
- Correctly encoding date and datetime fields.
- Converting currency amounts into numeric fields.
- Converting "Yes/No" flags to binary for modelling purposes.
- Re-naming columns for clarity and easy of use.

The 3 data files containing user, card and transaction information were joined together using the user id and card number fields.

The data was then split 80:20 into training and test data sets respectively.

One-hot encoding was used to convert some categorical data fields into binary variables for modelling.

All data was scaled according to the standard distributions of each variable.

### Dealing with class imbalance

A key stage of data preparation was to sub-sample the non-fraud transactions in the training data set to achieve a 50:50 class balance.

However, note that 99.88% of the transactions in the test set were non-fraud and so this represented the **"baseline"** against which model accuracy was assessed.

**INSERT PIC OF DATA SPLIT**

## Exploratory Data Analysis (EDA)

The most important aspect of EDA was looking at how fraud varied accross the distributions of each feature.  I formulated a consistent approach to doing this across categorical and continuous variables.  For both I created stacked bar charts.  For categoric variables each bar represented a category.  For continuous variables, the categories were formed by creating histogram bins.  The stacked bars were made up of the percentages of fraud and non-fraud transactions in that category/bin.  Over the top I plotted a line chart showig the number of samples in each category/bin so I could check by eye that any trend I could see was backed up by a statistically significant number of samples.

Here is an example.  This chart shows how fraud varies by hour of the day.  There is a greater chance of a transaction being fraudulent during the early hours of the morning or through the middle of the day.  Note how the left-hand scale starts at close to 100%.  That is because we are looking for relative differences in a very small proportion of transactions which are fraud.

**INSERT FRAUD BY HOUR OF DAY**

Other factors which make a transaction more likely to be fraudulent:-
- If it takes place online
- If it takes place abroad
- If it relates to certain types of retailers, particularly those onboard cruise liners, and music shops.  (This information is contained in a field called "Merchant Category Code").
- The value of the transaction - fraudulent transactions are bigger, averaging nearly \\$109 verses \\$44 for a non-fraud transaction.
- The age of the credit-card user - older users are more vulnerable.
- The annual income of the credit-card user.  Those on lower income are more vulnerable.
- The type of card.  Pre-paid debit cards are most vulnerable and the Discover brand is more vulnerable than Amex, Visa and Mastercard.

### Data correlations

Data correlations were reviewed by reference to heat maps.  Here's a picture showing the heat map of the standard data features.  The bottom row of the heatmap is the target variable, "tx_is_fraud".  As well as seeing many of the correlations observed in the EDA, some colinearity in the data is also evident and consequently some of these variables are "tuned out" by the models.

**INSERT HEATMAP OF CORRELATIONS IN STANDARD DATA**

## Feature engineering

Extra features were engineered into the data set to improve classifier performance.  These fell into 2 categories:
- Feaures which utilise information from previous transactions to put the next in context.  For example, a running average of the last 5 transaction amounts which the current transaction amount can be compared to.
- Features indicating high-fraud-rate values in categorical variables, e.g. "high_fraud_hour" which is set to "1" if the transaction took place in an hour of the day with above average rate of fraud.

** INCLUDE CODE SNIPPET HERE - engineered features **



## Limitations

Simulated data

## Need to cover
- Intro, contents, goals, hypothesis, tech & tools list
- Process/approach - step by step - include code snippets.
- Conclusions
- Wins & challenges:  What went well?  What did I struggle with?
- Key learnings:  What did you take away from the project?  How did it shape you as a Data Scientist?