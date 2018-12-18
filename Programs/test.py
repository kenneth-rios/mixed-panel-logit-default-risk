# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:48:25 2018

@author: kenrios
"""

import pylogit
import numpy as np
import pandas as pd

from collections import OrderedDict 

# Import training data
data = pd.read_excel("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Raw Data\\panel_train.xlsx")

# Initialize ordered dictionaries
basic_specification = OrderedDict()
basic_names = OrderedDict()


# Model year-fixed effects
basic_specification["year2008"] = [["Default"]]
basic_names["year2008"] = ["Year 2008"]

basic_specification["year2009"] = [["Default"]]
basic_names["year2009"] = ["Year 2009"]

basic_specification["year2010"] = [["Default"]]
basic_names["year2010"] = ["Year 2010"]

basic_specification["year2011"] = [["Default"]]
basic_names["year2011"] = ["Year 2011"]


# Model random coefficients over countries
basic_specification["gdp"] = [["Default"]]
basic_names["gdp"] = ["GDP"]

basic_specification["pci"] = [["Default"]]
basic_names["pci"] = ["Inflation"]


for var in ["gdp", "pci"]:
    data[var] = data[var].astype("float")
    

# Store variables whose coefficients are to be randomized over countries
index_var_names = ["GDP", "Inflation"]


# Create mixed logit model with year fixed-effects and random coefficients over countries
model = pylogit.create_choice_model(data = data,
                                   alt_id_col = "status",
                                   obs_id_col = "year",
                                   choice_col = "default",  # =1 for default, =0 for no default
                                   specification= basic_specification,
                                   model_type = "Mixed Logit",  # mixed panel logit model
                                   names = basic_names,
                                   mixing_id_col = "country",  # implies coefficients randomized over countries
                                   mixing_vars = index_var_names)

 
# Estimate mixed logit model using Nelder-Mead algorithm (cross-validated to choose optimal lambda)
model.fit_mle(init_vals = np.zeros(8),
              num_draws = 1000,  # 1000 draws from independent normal distributions for each parameter,
              seed = 3,          # as a function of its mean and standard deviation
              method = "Nelder-Mead",
              ridge = 1)  # ridge = penalty term on sum of squares of estimated parameters

model.log_likelihood

# Output estimation results
# Random coefficients are reported at their means over countries, drawn from normal distributions.
print("\n")
print(model.get_statsmodels_summary())  # P-values will definitely improve with more data!
                                        # Year-fixed effects will become important for recession years!


# Output predicted unconditional probabilities using estimated coefficients from mixed logit model on test data
counterfactual = pd.read_excel("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Raw Data\\panel_test.xlsx")

for var in ["gdp", "pci"]:
    counterfactual[var] = counterfactual[var].astype("float")
    
    
    
    
    
# Fit unconditional probabilities using panel_predict()
# Note that seemingly "unlikely" defaults in the training data are at the lower end of the probability range. 
# Take advantage of this gradient in probability to map to average haircuts!
data["probs"] = model.panel_predict(data,
                                   num_draws = 1000,
                                   seed = 2)





# Forecast unconditional probabilities using panel_predict()
counterfactual["probs"] = model.panel_predict(counterfactual,
                                             num_draws = 1000,  # Use 1000 draws from the estimated independent  
                                             seed = 2)          # normal distribution for each randomized coefficient


# Calculate and store log-likelihoods (use in k-fold cross-validation: for each possible lambda in the grid,
#                                       use k-1 folds to fit model and then predict on kth fold and calculate
#                                       the negative log-likelihood, then average over all folds).
observed_values = counterfactual["default"].values
log_predicted_probs = np.log(counterfactual["probs"])

negative_log_likelihood = -1 * observed_values.dot(log_predicted_probs)  # Choose lambda that returns lowest negative_log_likelihood
