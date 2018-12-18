# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 23:48:25 2018

@author: kenrios
"""

import os
import pandas as pd

# Change working directory to Data folder
os.chdir("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Data\\")

# Import LHS spreadsheet
LHS = pd.read_excel("external_debt_defaults_master.xlsx")
LHS["Status"] = "Default"

# Import EIU RHS data
EIU = pd.read_excel("EIU_data.xlsx")
EIU["Country"] = EIU["Country"].str.title()

# Import World Bank (WB) RHS data
WB = pd.read_excel("WB_data.xlsx")

# Inner-join LHS and RHS spreadsheets to contain a common set of observations
data = pd.merge(EIU, WB, how = "inner", on = ["Country", "Year"])
data = pd.merge(LHS, data, how = "inner", on = ["Country", "Year"])


# Subset for relavent RHS economic variables
# EIU
#EIU_vars = ["DGDP", "RYPC",
#            "CGEB", 
#            "PSBR", 
#            "DCPN",
#            "BINT", "PUDP",
#            "SODD",
#            "UNEM",
#            "CARA",
#            "IRTD",
#            "TDPX", "TDPY",
#            "TSPY",
#            "INPS",
#            "INPY"] 
#
# WB
#WB_vars = ['Totalreserves_pctoftotalexternal', 
#           'Totaldebtservice_pctofGNI', 
#           'Realeffectiveexchangerateindex_2',
#           'Presentvalueofexternaldebt_pctof',
#           'Netforeignassets_currentLCU',
#           'Interestpaymentsonexternaldebt_p',
#           'Inflationconsumerprices_annualpc',
#           'GNIpercapitagrowth_annualpct', 
#           'Externaldebtstocks_pctofGNI',
#           'Externalbalanceongoodsandservice', 
#           'Currentaccountbalance_BoPcurrent',
#           'Centralgovernmentdebttotal_pctof',
#           'Broadmoneytototalreservesratio']

# To Shukrit: You can manually add in the overlapping variables specified in the email
#             from WB into the WB_vars list below.
# EIU 
EIU_vars = ['DGDP', 'RYPC',
            'CGEB', 
            'PSBR', 
            'BINT', 'PUDP',
            'SODD',
            'CARA',
            'IRTD',
            'TDPX', 'TDPY',
            'TSPY',
            'INPS',
            'INPY',
            'XRRE'] 

# WB
WB_vars = ['Netforeignassets_currentLCU',
           'Inflationconsumerprices_annualpc',
           'Externalbalanceongoodsandservice', 
           'Currentaccountbalance_BoPcurrent',
           'Nettradeingoodsandservices_BoPcu',
           'Unemploymenttotal_pctoftotallabo']


data = data[["Country", "Year", "Status", "default_RR"] + WB_vars + EIU_vars]


# According to diagnostic, remove flagrant countries:
data = data[~data.Country.isin(['Austria',
                                'Antigua And Barbuda',
                                'Belgium',
                                'Canada',
                                'Cuba',
                                'Denmark',
                                'Finland',
                                'France',
                                'Germany',
                                'Grenada',
                                'Guinea-Bissau',
                                'Japan',
                                'Nauru',
                                'Netherlands',
                                'North Korea',
                                'Norway',
                                'Sweden',
                                'United Kingdom',
                                'United States'])]

# DIAGNOSTIC: By country, calculate number of NAs for each variable
check = data.isnull().groupby(data["Country"]).sum()

# Return dataset for countries with at least one '35' across variables
check35s = check.loc[check.apply(lambda row: row.astype(str).str.contains("35").any(), axis=1), ]


# Replace all economic variables with their one-year lags
data.iloc[:, 4:25] = data.groupby(data["Country"]).shift(1).iloc[:, 3:24]

# Purge data set of any observations with missing data
data = data.dropna(axis=0)  # To Shukrit: Leave this line commented out to inspect defaults with missing values!
#data["missing"] = data.iloc[:, 4:25].isnull().sum(axis = 1)


# Export mixed panel data as .csv
data.to_csv("C:\\Users\\kenri\\Data_Bootcamp\\Research Project\\Python\\Output\\mixed_panel_data.csv", index = False)
