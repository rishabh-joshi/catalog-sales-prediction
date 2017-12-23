# Catalog Sales Prediction

## Executive Summary
To increase the return on investment (ROI) of mailed catalogs, we analyzed sales data from existing customers who received catalogs to identify high value targets for our marketing efforts in the future. Our analysis discovered the following key predictors for determining whether customers would be likely to respond to a catalog with a purchase: consistency of sales in consecutive years, recency of last purchases, the activeness of a customer, sales within the past year, and their average spend per order. The combination of our final logistic regression and multiple linear regression models resulted in a payoff of $51,465.45. In theory, a perfect model would result in a payoff of $120,252.40 from the test set. While not perfect, our model still has predicting power as seen through its predictions capturing 42.80% of the maximum payoff.

## Business Situation

The file `catalog sales data.csv` comes from a retail company that sells upscale clothing on its website and via catalogs, which help drive customers to the website. All customers were sent a catalog mailing on Sep 1, 2012. On Dec 1, 2012 it was recorded whether or not they responded by making a purchase. There is one row for each customer. The `targdol` is the response variable, which is the purchase amount in response to receiving the catalog (`targdol` = 0 indicates that the customer did not respond). The remainder of variables are potential predictor variables which give information about the customer as of the time of the mailing. LTD means "life-to-date," i.e. since the customer purchased for the rst time.

## Data Dictionary

There are a total 101,532 customers, who are randomly split into 50418 in the training set and the remaining 51,114 in the test set (train =1 training set, train =0 test set). The definitions of the variables are as follows.

- targdol: dollar purchase resulting from catalog mailing
- datead6: date added to file
- datelp6: date of last purchase
- lpuryear: latest purchase year
- slstyr: sales ($) this year
- slslyr: sales ($) last year
- sls2ago: sales ($) 2 years ago
- sls3ago: sales ($) 3 years ago
- slshist: LTD dollars
- ordtyr: orders this year
- ordlyr: orders last year
- ord2ago: orders 2 years ago
- ord3ago: orders 3 years ago
- ordhist: LTD orders
- falord: LTD fall orders
- sprord: LTD spring orders
- train: training/test set indicator (1 = training, 0 = test)

## Goal

Build a predictive model for `targdol` based on the training set and then test it on the test set.

## Description of the `.Rmd` files

The `.Rmd` files should be exeuted in the following order:

1. `data_cleaning.Rmd` - Generates two data files called `clean_train.csv` and  `clean_test.csv` containing the preprocessed training and testing data sets respectively.

2. `logistic_modeling.Rmd` - Fits multiple logistic regression models and chooses the best model. Generates the probability of responding for each customer in the file `probabilities.csv`.

3. `linear_modeling.Rmd` - Fits multiple linear regression models and chooses the best model. Combines the logistic and linear models to compute the payoff and MSPE

## Criteria for Evaluating the Fitted Models:

- **General:** The final fitted regression model should meet the usual criteria such as significant coefficients, satisfactory residual plots, good fit as measured by $R^2$ or $R^2$ adjusted, parsimony and interpretability of the model etc.
- **Financial Criterion:** Select the top 1000 customers (prospects) from the test set who have the highest E(targdol). Then nd their total actual purchases. This is the payoff and should be as high as possible.
