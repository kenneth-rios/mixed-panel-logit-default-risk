# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 01:02:43 2018

@author: kenri
"""

import pylogit
import numpy as np
import pandas as pd

from collections import OrderedDict 

data = pd.read_excel("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Raw Data\\mock_conditional_logit_long_data.xlsx")
counterfactual = data

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["intercept"] = ["A", "B", "C", "D"]
basic_names["intercept"] = ["A", "B", "C", "D"]


basic_specification["year2008"] = [["A", "B", "C", "D"]]
basic_names["year2008"] = ["Year 2008"]

basic_specification["year2009"] = [["A", "B", "C", "D"]]
basic_names["year2009"] = ["Year 2009"]

basic_specification["year2010"] = [["A", "B", "C", "D"]]
basic_names["year2010"] = ["Year 2010"]

# Remove Year 2011 due to collinearity!
#basic_specification["year2011"] = [["A", "B", "C", "D"]]
#basic_names["year2011"] = ["Year 2011"]


basic_specification["gdp"] = [["A", "B", "C", "D"]]
basic_names["gdp"] = ["GDP"]

basic_specification["pci"] = [["A", "B", "C", "D"]]
basic_names["pci"] = ["Inflation"]



# Create conditional logit model with year fixed-effects
model = pylogit.create_choice_model(data=data,
                                alt_id_col= "country",
                                obs_id_col= "year",
                                choice_col= "default",
                                specification = basic_specification,
                                model_type = "MNL",
                                names = basic_names)


# Estimate conditional logit model using BFGS algorithm
model.fit_mle(np.zeros(9), method = "BFGS", ridge = .00005)

# Output estimation results
model.get_statsmodels_summary()


# Output predicted probabilities using estimated coefficients from conditional logit model
counterfactual["prob"] = model.predict(counterfactual)



