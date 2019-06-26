# -*- coding: utf-8 -*-

"""
Created on Sat Jun  1 03:58:48 2019

@author: jubin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.plotly as py


ac50 = pd.read_csv("D:/INVITRODB_V3_1_SUMMARY/ac50_Matrix_190226.csv")

ac50_tran = ac50.melt(id_vars=["Unnamed: 0"],
                      var_name="TestName",
                      value_name="EC50")

# suppressing the scientific notation e

ac50_tran['EC50'] = ac50_tran['EC50'].apply(lambda x: '{:.4f}'.format(x))


# ac50_tran.dropna(how='any', inplace = True)

ac50_tran = ac50_tran[~(ac50_tran["EC50"]=='nan')]

# Working Cmax data frame
cmax = pd.read_excel(open("D:/INVITRODB_V3_1_SUMMARY/500_Cmax_Project.xlsx",'rb'
                          ),
                     sheetname ='Sheet1')

cmax.columns
cmax = cmax.rename(columns={"CmaxStand_μM)": "Cmax"})

ac50_tran = cmax.rename(columns={"Un": "Cmax"})

cmax['Cmax'].isnull().sum()
cmax['code'].nunique()

cmax_new = cmax[~(cmax["Cmax"]=='nan')]
cmax = cmax.drop('Ref_PMID',axis=1)

cmax['Cmax'].dtype
cmax = cmax.dropna(how='any',axis=0) 

# renaming ac5-_tran data frame
ac50_tran = ac50_tran.rename(columns={"Unnamed: 0": "code"})

# merging two files:
final = pd.merge(ac50_tran, cmax, how='inner', on=['code'])

final.isnull().values.any()

# shuffling the data set 
from sklearn.utils import shuffle
final_med_dili = shuffle(final)

# Dropping columns:
final_mednas = final_med_dili.drop(final_med_dili.columns[[4,5]], axis=1)

le = len(final_mednas['EC50'])


for i in range(0,le-1):
    if final_mednas['EC50'].values[i] == '1000000.0000':
        final_mednas['EC50'].values[i] = '0'
        
        
# extracting the csv file
final_mednas.to_csv("D:/INVITRODB_V3_1_SUMMARY/JM_med_dili.csv")

med_nas = pd.read_csv("D:/INVITRODB_V3_1_SUMMARY/JM_med_dili.csv")

med_nas = med_nas.drop(med_nas.columns[[0]], axis = 1)












