# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 01:02:43 2018

@author: kenri
"""

import pylogit
import numpy as np
import pandas as pd

from collections import OrderedDict 

data = pd.read_excel("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Raw Data\\train2.xlsx")

basic_specification = OrderedDict()
basic_names = OrderedDict()


#basic_specification["A"] = [[1, 2, 3, 4]]
#basic_names["A"] = ["A"]
#
#basic_specification["B"] = [[1, 2, 3, 4]]
#basic_names["B"] = ["B"]
#
#basic_specification["C"] = [[1, 2, 3, 4]]
#basic_names["C"] = ["C"]
#
#basic_specification["D"] = [[1, 2, 3, 4]]
#basic_names["D"] = ["D"]

basic_specification["year2008"] = [[2008, 2009, 2010, 2011]]
basic_names["year2008"] = ["Year 2008"]

basic_specification["year2009"] = [[2008, 2009, 2010, 2011]]
basic_names["year2009"] = ["Year 2009"]

basic_specification["year2010"] = [[2008, 2009, 2010, 2011]]
basic_names["year2010"] = ["Year 2010"]

basic_specification["year2011"] = [[2008, 2009, 2010, 2011]]
basic_names["year2011"] = ["Year 2011"]


basic_specification["gdp"] = [2008, 2009, 2010, 2011]
basic_names["gdp"] = ["GDP 1", "GDP 2", "GDP 3", "GDP 4"]

basic_specification["pci"] = [2008, 2009, 2010, 2011]
basic_names["pci"] = ["Inflation 1", "Inflation 2", "Inflation 3", "Inflation 4"]

#basic_specification["gdp"] = [[1, 2, 3, 4]]
#basic_names["gdp"] = ["GDP"]
#
#basic_specification["pci"] = [[1, 2, 3, 4]]
#basic_names["pci"] = ["Inflation"]


for var in ["gdp", "pci"]:
    data[var] = data[var].astype("float")

index_var_names = ["GDP 1", "GDP 2", "GDP 3", "GDP 4",
                   "Inflation 1", "Inflation 2", "Inflation 3", "Inflation 4"]

# Create mixed logit model with year fixed-effects and random country effects
model = pylogit.create_choice_model(data = data,
                                   alt_id_col = "year",
                                   obs_id_col = "country",
                                   choice_col = "default",
                                   specification= basic_specification,
                                   model_type = "Mixed Logit",
                                   names = basic_names,
                                   mixing_id_col = "country",  # implies country random effects
                                   mixing_vars = index_var_names)

# Estimate mixed logit model using Nelder-Mead algorithm
model.fit_mle(init_vals = np.zeros(20),
              num_draws = 1000,  # 1000 draws from the normal distribution
              seed = 3,
              method = "Nelder-Mead",
              ridge = 1000)  # ridge = penalty term on sum of squares of estimated coefficients

# Output estimation results
print("\n")
print(model.get_statsmodels_summary())


# Output predicted unconditional probabilities using estimated coefficients from mixed logit model on test data
counterfactual = pd.read_excel("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Raw Data\\test2.xlsx")

for var in ["gdp", "pci"]:
    counterfactual[var] = counterfactual[var].astype("float")
    
counterfactual["prob"] = model.panel_predict(counterfactual,
                                             num_draws = 1000,
                                             seed = 2)
