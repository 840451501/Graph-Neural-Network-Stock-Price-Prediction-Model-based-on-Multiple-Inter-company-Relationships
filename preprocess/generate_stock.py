# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:56:58 2023

@author: Administrator
"""

from datetime import datetime
import json
import numpy as np
import operator
import os
import pandas as pd


tickers1 = pd.read_excel('../data/astock/06-08_1.xlsx', index_col=None, sheet_name = 0)
tickers2 = pd.read_excel('../data/astock/06-08_2.xlsx', index_col=None, sheet_name = 0)
tickers3 = pd.read_excel('../data/astock/09-13_1.xlsx', index_col=None, sheet_name = 0)
tickers4 = pd.read_excel('../data/astock/09-13_2.xlsx', index_col=None, sheet_name = 0)
tickers5 = pd.read_excel('../data/astock/09-13_3.xlsx', index_col=None, sheet_name = 0)
tickers6 = pd.read_excel('../data/astock/14-18_1.xlsx', index_col=None, sheet_name = 0)
tickers7 = pd.read_excel('../data/astock/14-18_2.xlsx', index_col=None, sheet_name = 0)
tickers8 = pd.read_excel('../data/astock/14-18_3.xlsx', index_col=None, sheet_name = 0)
tickers9 = pd.read_excel('../data/astock/14-18_4.xlsx', index_col=None, sheet_name = 0)
tickers10 = pd.read_excel('../data/astock/18-23_1.xlsx', index_col=None, sheet_name = 0)
tickers11 = pd.read_excel('../data/astock/18-23_2.xlsx', index_col=None, sheet_name = 0)
tickers12 = pd.read_excel('../data/astock/18-23_3.xlsx', index_col=None, sheet_name = 0)
tickers13 = pd.read_excel('../data/astock/18-23_4.xlsx', index_col=None, sheet_name = 0)
tickers14 = pd.read_excel('../data/astock/18-23_5.xlsx', index_col=None, sheet_name = 0)
tickers15 = pd.read_excel('../data/astock/18-23_6.xlsx', index_col=None, sheet_name = 0)

tickers_all = pd.concat([tickers1, tickers2, tickers3, tickers4, tickers5, tickers6, tickers7, tickers8, tickers9, tickers10, tickers11, tickers12, tickers13, tickers14, tickers15])

for i in range(0, tickers_all.shape[0]):
    if (tickers_all.iloc[i, 0][0] == '0') or (tickers_all.iloc[i, 0][0] == '3'):
        tickers_all.iloc[i, 0] = tickers_all.iloc[i, 0] + '.SZ'
    if tickers_all.iloc[i, 0][0] == '6':
        tickers_all.iloc[i, 0] = tickers_all.iloc[i, 0] + '.SH'
    if (tickers_all.iloc[i, 0][0] == '8') or (tickers_all.iloc[i, 0][0] == '4'):
        tickers_all.iloc[i, 0] = tickers_all.iloc[i, 0] + '.BJ'



#tickers_all.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)
tickers_all.columns = tickers_all.iloc[0, :]

loc_1 = list(tickers_all.loc[tickers_all['证券代码'] == '证券代码'].index)
loc_2 = list(tickers_all.loc[tickers_all['证券代码'] == '没有单位'].index)
tickers_all.drop(index = loc_1 + loc_2, axis = 0, inplace = True)
tickers_all.sort_values("证券代码", inplace=True)
tickers_all.to_csv('../data/astock/stock_data.csv')