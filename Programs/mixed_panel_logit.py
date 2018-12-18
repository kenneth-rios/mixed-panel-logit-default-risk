# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:48:25 2018

@author: kenrios
"""

import pylogit  # must download source files @ https://github.com/timothyb0912/pylogit
import numpy as np
import pandas as pd

from collections import OrderedDict 

# Import data
train = pd.read_csv("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Output\\mixed_panel_data.csv")

# Generate year dummies for each year present in the data
year_dummies = pd.get_dummies(train["Year"]).rename(columns = lambda x: 'year' + str(x))
train = pd.concat([train, year_dummies], axis=1)


# Subset training and tests datasets according to cutoff year
test = train.loc[train.Year >= 2010]          # Post-2009
train = train[~train.index.isin(test.index)]  # Pre-2010

set(train["Country"].unique()).symmetric_difference(set(test["Country"].unique()))

# Initialize ordered dictionaries
basic_specification = OrderedDict()
basic_names = OrderedDict()


# Model year-fixed effects (possibly only include recession years in order to ensure convergence!)
#for year in year_dummies.columns.tolist():
#    basic_specification[year] = [["Default"]]
#    basic_names[year] = ["Year " + year[-4:]]

basic_specification["year2001"] = [["Default"]]
basic_names["year2001"] = ["Year 2001"]

basic_specification["year2007"] = [["Default"]]
basic_names["year2007"] = ["Year 2007"]

basic_specification["year2008"] = [["Default"]]
basic_names["year2008"] = ["Year 2008"]

basic_specification["year2009"] = [["Default"]]
basic_names["year2009"] = ["Year 2009"]

    
# Model coefficients randomized over countries
vars = {"Netforeignassets_currentLCU" : "2",
        "Inflationconsumerprices_annualpc" : "Inflation",
        "Externalbalanceongoodsandservice" : "thre",
        "Currentaccountbalance_BoPcurrent" : "thrgf",
        "Nettradeingoodsandservices_BoPcu" : "threfgdf",
        "Unemploymenttotal_pctoftotallabo" : "thredsfe",
        "DGDP" : "thdf",
        "RYPC" : "thsgdbe",
        "CGEB" : "thfe",
        "PSBR" : "thgdfgee",
        "BINT" : "threfe",
        "PUDP" : "thr4e",
        "SODD" : "thr6e",
        "CARA" : "thr3e",
        "IRTD" : "thre",
        "TDPY" : "three",
        "TDPX" : "three",
        "TSPY" : "three",
        "INPS" : "three",
        "INPY" : "three",
        "XRRE" : "Exchange Rate"}

for key in vars:
    basic_specification[key] = [["Default"]]
    basic_names[key] = [vars[key]]

# Store variables whose coefficients are to be randomized over countries
index_var_names = list(vars.values())
    




##### 10-FOLD CROSS-VALIDATION #####
print("\n##### 10-FOLD CROSS-VALIDATION #####\n")

# Initialize lambda dictionary
lambdas = {}

# Shuffle training set
train = train.sample(frac = 1, random_state = 3)

# Generate a list of 10 cutoff indices that equally divide the training data, to be automated later
indices = [0, 65, 130, 195, 260, 325, 390, 455, 520, 585, 650]

for i in np.arange(0, 100, 1):
#for i in [30]:
    
    NLL = 0
    
    for k in range(len(indices)-1):

        # For each fold k, create the holdout set and the bagged set
        holdout = train.iloc[indices[k]:indices[k+1], :].sort_values(by = ["Year", "Country"])
        bagged = train[~train.index.isin(holdout.index)].sort_values(by = ["Year", "Country"])
    
        # Create mixed logit model with year fixed-effects and random coefficients over countries
        model = pylogit.create_choice_model(data = bagged,
                                            alt_id_col = "Status",
                                            obs_id_col = "Year",
                                            choice_col = "default_RR",   # =1 for default, =0 for no default
                                            specification= basic_specification,
                                            model_type = "Mixed Logit",  # mixed panel logit model
                                            names = basic_names,
                                            mixing_id_col = "Country",   # implies coefficients randomized over countries
                                            mixing_vars = index_var_names)
        
        
        # Estimate mixed logit model using Nelder-Mead algorithm (cross-validated to choose optimal lambda) on K-1 folds
        model.fit_mle(init_vals = np.zeros(46),
                      num_draws = 1000,  # 1000 draws from independent normal distributions for each parameter,
                      #seed = 1,         # as functions of their means and standard deviations
                      method = "Nelder-Mead",
                      maxiter = 10,  # number of Nelder-Mead iterations
                      ridge = i)     # ridge = penalty term 'i' on the sum of squares of estimated parameters
        
        
        # Forecast unconditional probabilities using panel_predict() on the kth fold
        probs = model.panel_predict(holdout,
                                    #seed = 1
                                    num_draws = 1000)  # use 1000 draws from the estimated independent  
                                                       # normal distribution for each randomized coefficient
        
        
        # Calculate negative log-likelihood, which is the cross-validation error of the kth fold
        observed_values = holdout["default_RR"].values
        log_predicted_probs = np.log(probs)
        
        negative_log_likelihood = -1 * observed_values.dot(log_predicted_probs)
        NLL += negative_log_likelihood
    
        print("\n")
        print("ESTIMATION FOR FOLD " + str(k+1) + " USING LAMBDA = " + str(i) + " CONVERGED")
        print("\n")
    
    # Calculate CV negative log-likelihood for given lambda and store in lambdas dictionary
    CV_NLL = NLL / 10
    print("THE CROSS-VALIDATION ERROR FOR LAMBDA = " + str(i) + " IS " + str(CV_NLL))
    print("\n")
    
    lambdas[i] = CV_NLL
        
# Return lambda which corresponds to the -lowest- CV negative log-likelihood
lambda_CV = sorted(lambdas, key = lambdas.get)[0]




#### PREDICTION #####
print("\n##### OUT-OF-SAMPLE PREDICTION #####\n")

# Fit the mixed panel logit model on the entire training dataset using the optimal lambda_CV
model = pylogit.create_choice_model(data = train,
                                    alt_id_col = "Status",
                                    obs_id_col = "Year",
                                    choice_col = "default_RR",   
                                    specification= basic_specification,
                                    model_type = "Mixed Logit",  
                                    names = basic_names,
                                    mixing_id_col = "Country",  
                                    mixing_vars = index_var_names)

model.fit_mle(init_vals = np.zeros(46),
              num_draws = 1000,
              #seed = 1,         
              method = "Nelder-Mead",
              maxiter = 10,  
              ridge = lambda_CV)     


# Predict unconditional probabilities on the test set using the fully calibrated model
test["probs"] = model.panel_predict(test,
                                    #seed = 1
                                    num_draws = 1000)
                                         










## Create mixed logit model with year fixed-effects and random coefficients over countries
#model = pylogit.create_choice_model(data = data,
#                                   alt_id_col = "status",
#                                   obs_id_col = "year",
#                                   choice_col = "default",  # =1 for default, =0 for no default
#                                   specification= basic_specification,
#                                   model_type = "Mixed Logit",  # mixed panel logit model
#                                   names = basic_names,
#                                   mixing_id_col = "country",  # implies coefficients randomized over countries
#                                   mixing_vars = index_var_names)
#
# 
## Estimate mixed logit model using Nelder-Mead algorithm (cross-validated to choose optimal lambda)
#model.fit_mle(init_vals = np.zeros(8),
#              num_draws = 1000,  # 1000 draws from independent normal distributions for each parameter,
#              seed = 3,          # as a function of its mean and standard deviation
#              method = "Nelder-Mead",
#              ridge = 1)  # ridge = penalty term on sum of squares of estimated parameters
#
#model.log_likelihood
#
## Output estimation results
## Random coefficients are reported at their means over countries, drawn from normal distributions.
#print("\n")
#print(model.get_statsmodels_summary())  # P-values will definitely improve with more data!
#                                        # Year-fixed effects will become important for recession years!
#
#
## Output predicted unconditional probabilities using estimated coefficients from mixed logit model on test data
#counterfactual = pd.read_excel("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Raw Data\\panel_test.xlsx")
#
#for var in ["gdp", "pci"]:
#    counterfactual[var] = counterfactual[var].astype("float")
#    
#    
#    
#    
#    
## Fit unconditional probabilities using panel_predict()
## Note that seemingly "unlikely" defaults in the training data are at the lower end of the probability range. 
## Take advantage of this gradient in probability to map to average haircuts!
#data["probs"] = model.panel_predict(data,
#                                   num_draws = 1000,
#                                   seed = 2)
#
#
#
#
#
## Forecast unconditional probabilities using panel_predict()
#counterfactual["probs"] = model.panel_predict(counterfactual,
#                                             num_draws = 1000,  # Use 1000 draws from the estimated independent  
#                                             seed = 2)          # normal distribution for each randomized coefficient
#
#
## Calculate and store log-likelihoods (use in k-fold cross-validation: for each possible lambda in the grid,
##                                       use k-1 folds to fit model and then predict on kth fold and calculate
##                                       the negative log-likelihood, then average over all folds).
#observed_values = counterfactual["default"].values
#log_predicted_probs = np.log(counterfactual["probs"])
#
#negative_log_likelihood = -1 * observed_values.dot(log_predicted_probs)  # Choose lambda that returns lowest negative_log_likelihood
