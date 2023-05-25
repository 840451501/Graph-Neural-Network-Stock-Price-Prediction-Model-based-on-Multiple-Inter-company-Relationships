# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:13:34 2023

@author: 84045
"""
#关系数据预处理
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import re

def name_process(stock_list):
    stock_match_name = pd.DataFrame(index = range(0, stock_num), columns = ['简称', '全称', '两字'])
    #信息和股票是否对应
    for i in range(0,stock_num):
        simple_name = stock_list.iloc[i,1]
        full_name = stock_list.iloc[i,2]
        simple_name = re.sub('[a-zA-Z]','',simple_name)
        stock_match_name.iloc[i, 0] = re.sub('\-','',simple_name)
        full_name = re.sub('股份有限公司','',full_name)
        tmp_index = re.search('\(',full_name)
        if tmp_index == None:
            stock_match_name.iloc[i, 1] = full_name
        else:
            tmp_index = tmp_index.span()[0]
            stock_match_name.iloc[i, 1] = full_name[:tmp_index]
        if len(stock_match_name.iloc[i, 0])>3:
            stock_match_name.iloc[i, 2] = stock_match_name.iloc[i, 0][0:2]
    return stock_match_name

def gongying_test(stock):
    '''
    #提取公告数据
    tmp_loc = stock=='潜在客户'
    info_index = tmp_loc.idxmax()
    series_index = list(stock.index)
    index = series_index.index(info_index)+1
    info = stock[series_index[index:-1]]
    info=[info.iloc[x] for x in range(1,len(info),2)]
    
    #判断信息匹配
    full_stock = ''
    for j in range(0, len(info)):
        full_stock = full_stock + str(info[j])
    simple_name = stock_match_name.iloc[i, 0]
    full_name = stock_match_name.iloc[i, 1]
    
    if (label == 0) and (stock_match_name.iloc[i, 0] not in full_stock) and (stock_match_name.iloc[i, 1] not in full_stock):
        wrong_dict[i] = [stock_list.iloc[i, 2]]
        continue
    '''
    '''
    #用来判断是否和上一个一样
    if i != 0:
        if len(stock_last) == len(stock):
            tmp_same = stock_last == stock
            if sum(tmp_same) == stock_last.shape[0]:
                print(stock_list['搜索名称'][i]+' 信息重复')
            stock_last = stock
    '''
    '''
    #判断是否需要翻页
    if stock.iloc[-1] == 2:
        print(stock_match_name.iloc[i, 1]+' 需要翻页')
    '''

def company_match(full_stock, stock_match_name, i):
    match_loc = []
    for j in range(0, stock_match_name.shape[0]):
        if j == i:
            continue
        else:
            simple_name = stock_match_name.iloc[j, 0]
            full_name = stock_match_name.iloc[j, 1]
            if (simple_name in full_stock) or (full_name in full_stock):
                match_loc.append(j)
    return match_loc

rel_number = 3

stock_list = pd.read_excel('../data/公司名单_供应.xlsx', index_col=None, sheet_name = 0)
stock_num = stock_list.shape[0]

#stock_match_name = name_process(stock_list)
stock_match_name = pd.read_excel('名称映射.xlsx', index_col=None)

wiki_relation_embedding = np.zeros(
    [stock_list.shape[0], stock_list.shape[0], rel_number + 1],#这里关系总数加1,是为了最后一个一定要为1
    dtype=int)
#wiki_relation_embedding = np.load('竞争.npy')
#wrong_dict = {}

#做独特矩阵，要把单列数据处理和导入关系结合起来做，不然不好存
#竞争关系
raw_data = pd.read_excel('../data/公司名单_投资.xlsx', index_col=None, sheet_name = 1)
stock_last = raw_data.iloc[:,0]
for i in range(0,stock_num):#这里10改成stock_num
#融资
    label = 0
    stock = raw_data.iloc[:,i]
    stock = stock.dropna(how = 'all')
    tmp_loc = stock=='自然人'
    info_index = tmp_loc.idxmax()
    series_index = list(stock.index)
    index = series_index.index(info_index)+1
    info = stock[series_index[index:]]
    info_word = [info.iloc[x] for x in range(1,len(info),2)]
    info_feature = [info.iloc[x-1] for x in range(1,len(info),2)]
    
    print(i)
    
    for j in range(0, len(info_feature)):
        if '投资PEVC融资' in info_feature[j]:
            custom_list = company_match(info_word[j], stock_match_name, i)
            for p in custom_list:
                wiki_relation_embedding[i, p, 2] = 1
        '''
        if '投资并购重组' in info_feature[j]:
            custom_list = company_match(info_word[j], stock_match_name, i)
            for p in custom_list:
                wiki_relation_embedding[i, p, 3] = 1
                
        if '融资并购重组' in info_feature[j]:
            custom_list = company_match(info_word[j], stock_match_name, i)
            for p in custom_list:
                wiki_relation_embedding[i, p, 4] = 1

        if '投资银行借款' in info_feature[j]:
            custom_list = company_match(info_word[j], stock_match_name, i)
            for p in custom_list:
                wiki_relation_embedding[i, p, 5] = 1
        '''
    
#供应
raw_data = pd.read_excel('../data/公司名单_供应.xlsx', index_col=None, sheet_name = 1)
stock_last = raw_data.iloc[:,0]
for i in range(0,stock_num):#这里10改成stock_num
#供应
    #这里开始写进循环里
    label = 0
    stock = raw_data.iloc[:,i]
    stock = stock.dropna(how = 'all')

    #判断是否为空
    if stock.iloc[-1] == '对不起，数据为空' or stock.iloc[-1] == '暂无数据':
        label = 1
        #continue
    
    #提取公告数据
    tmp_loc = stock=='潜在客户'
    info_index = tmp_loc.idxmax()
    series_index = list(stock.index)
    index = series_index.index(info_index)+1
    info = stock[series_index[index:-1]]
    info_tmp = [info.iloc[x] for x in range(1,len(info),2)]
    
    full_stock = ''
    for j in range(0, len(info_tmp)):
        full_stock = full_stock + str(info_tmp[j])
        
    custom_list = company_match(full_stock, stock_match_name, i)
    for p in custom_list:
        wiki_relation_embedding[i, p, 1] = 1


#竞争
raw_data = pd.read_excel('../data/公司名单_竞争.xlsx', index_col=None, sheet_name = 1)
stock_last = raw_data.iloc[:,0]

for i in range(0,stock_num):#这里10改成stock_num
    stock = raw_data.iloc[:,i]
    stock = stock.dropna(how = 'all')

#    if stock[0] != stock_list['搜索名称'][i]:
#       print(stock_list['搜索名称'][i])
#       continue
    tmp_loc = stock=='基础信息'
    compete_company = stock[2:(tmp_loc.idxmax()-2)]
    
    for j in range(0,compete_company.shape[0]):
        competitor = compete_company.iloc[j]
        tmp_loc = stock_list['搜索名称简称'] == competitor
        if tmp_loc.sum():
            competitor_index = tmp_loc.idxmax()
            wiki_relation_embedding[i, competitor_index, 0] = 1
            wiki_relation_embedding[competitor_index, i, 0] = 1
        else:
            continue

for i in range(stock_list.shape[0]):
    wiki_relation_embedding[i, i, -1] = 1
#print(wiki_relation_embedding.shape)
np.save('汇总', wiki_relation_embedding)
