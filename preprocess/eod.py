import argparse
from datetime import datetime
import json
import numpy as np
import operator
import os
import pandas as pd

pd.set_option('mode.chained_assignment', None)

class EOD_Preprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d'

    def _read_EOD_data(self):
        self.data_EOD = []
        full_EOD = pd.read_csv('../data/astock/stock_data.csv', index_col=None)
        full_EOD.drop(columns = ['Unnamed: 0'], axis = 1, inplace = True)
        for i in range(0, self.tickers.shape[0]):
            '''
            single_EOD = np.genfromtxt(
                os.path.join(self.data_path, ticker +
                             '.xlsx'), dtype=str, delimiter=',',
                skip_header=True
            )
            '''

            index = full_EOD.loc[full_EOD['证券代码'] == self.tickers[i]].index
            stock = full_EOD.iloc[index,  :]
            stock.sort_values(by="交易日期" , inplace=True, ascending=True)
            stock.drop(columns = ['证券代码'], axis = 1, inplace = True)
            stock = stock[['交易日期', '日开盘价', '日最高价', '日最低价', '日收盘价', '日个股交易股数']]
            #stock = stock[]
            self.data_EOD.append(np.array(stock))
            # if index > 99:
            #     break
        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def _read_tickers(self, ticker_fname):
        self.tickers = np.genfromtxt(ticker_fname, dtype=str, delimiter='\t',
                                     skip_header=True)[:, 0]#这里是要自己改的，用xlsx来替代

    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        for row, daily_EOD in enumerate(selected_EOD_str):
            date_str = daily_EOD[0]
            #date_str = datetime.strptime(date_str, self.date_format)
            selected_EOD[row][0] = tra_date_index[date_str]
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row][col] = float(daily_EOD[col])
        return selected_EOD

    '''
        Transform the original EOD data collected from Google Finance to a
        friendly format to fit machine learning model via the following steps:
            Calculate moving average (5-days, 10-days, 20-days, 30-days),
            ignoring suspension days (market open, only suspend this stock)
            Normalize features by (feature - min) / (max - min)
    '''
    def generate_feature(self, begin_date, opath,
                         return_days=1, pad_begin=29):
        
        trading_dates = pd.read_excel('../data/trade_day.xlsx')
        trading_dates = trading_dates['日期']
        print('#trading dates:', len(trading_dates))
        # begin_date = datetime.strptime(trading_dates[29], self.date_format)
        print('begin date:', begin_date)
        # transform the trading dates into a dictionary with index
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            date = datetime.strftime(date, self.date_format)
            tra_dates_index[date] = index
            index_tra_dates[index] = date
        
        self.tickers = pd.read_excel('../data/公司名单_供应.xlsx', index_col=None, sheet_name = 0)
        self.tickers = self.tickers['代码'].values
        '''
        self.tickers = np.genfromtxt(
            os.path.join(self.data_path, 'trade_list.xlsx'),
            dtype=str, delimiter='\t', skip_header=False
        )                                                  #这个很重要，是股票列表名单，要改
        '''
        
        print('#tickers selected:', len(self.tickers))
        #self._read_EOD_data()
        
        #np.save('../data/data_EOD.npy', self.data_EOD)
        self.data_EOD = np.load("../data/data_EOD.npy", allow_pickle=True)
        
        wrong_index = []
        for stock_index, single_EOD in enumerate(self.data_EOD):
            
            #stock_index = 4419
            #single_EOD = self.data_EOD[4830]
            
            # select data within the begin_date
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0]#.replace('-05:00', '')#这里怀疑要换成np，好像也不用，试一下
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date:
                    begin_date_row = date_index
                    break
            selected_EOD_str = single_EOD[begin_date_row:]
            selected_EOD = self._transfer_EOD_str(selected_EOD_str,
                                                  tra_dates_index)

            # calculate moving average features
            begin_date_row = -1
            for row in selected_EOD[:, 0]:
                row = int(row)
                if row >= pad_begin:   # offset for the first 30-days average，这里是为了计算后面30日平均
                    begin_date_row = row
                    break
            mov_aver_features = np.zeros(
                [selected_EOD.shape[0], 4], dtype=float
            )   # 4 columns refers to 5-, 10-, 20-, 30-days average
            for row in range(begin_date_row, selected_EOD.shape[0]):
                date_index = selected_EOD[row][0]
                aver_5 = 0.0
                aver_10 = 0.0
                aver_20 = 0.0
                aver_30 = 0.0
                count_5 = 0
                count_10 = 0
                count_20 = 0
                count_30 = 0
                for offset in range(30):
                    date_gap = date_index - selected_EOD[row - offset][0]
                    if date_gap < 5:
                        count_5 += 1
                        aver_5 += selected_EOD[row - offset][4]#移动平均拿收盘价算
                    if date_gap < 10:
                        count_10 += 1
                        aver_10 += selected_EOD[row - offset][4]
                    if date_gap < 20:
                        count_20 += 1
                        aver_20 += selected_EOD[row - offset][4]
                    if date_gap < 30:
                        count_30 += 1
                        aver_30 += selected_EOD[row - offset][4]
                mov_aver_features[row][0] = aver_5 / count_5
                mov_aver_features[row][1] = aver_10 / count_10
                mov_aver_features[row][2] = aver_20 / count_20
                mov_aver_features[row][3] = aver_30 / count_30

            '''
                normalize features by feature / max, the max price is the
                max of close prices, I give up to subtract min for easier
                return ratio calculation.
            '''
            try:
                price_max = np.max(selected_EOD[begin_date_row:, 4])
                # open_high_low = (selected_EOD[:, 1:4] - price_min) / \
                #                 (price_max - price_min)
                mov_aver_features = mov_aver_features / price_max#这里非常奇怪，把移动平均的除了最大股价，可能是为了归一化
    
                '''
                    generate feature and ground truth in the following format:
                    date_index, 5-day, 10-day, 20-day, 30-day, close price
                    two ways to pad missing dates:
                    for dates without record, pad a row [date_index, -1234 * 5]
                '''
                features = np.ones([len(trading_dates) - pad_begin, 6],
                                   dtype=float) * -1234
                # data missed at the beginning
                for row in range(len(trading_dates) - pad_begin):
                    features[row][0] = row#row，也就是第一列是行号，index
                for row in range(begin_date_row, selected_EOD.shape[0]):
                    cur_index = int(selected_EOD[row][0])#这里就是让日期从选定日期开始，选定日期跟offset有关
                    features[cur_index - pad_begin][1:5] = mov_aver_features[
                        row]
                    if cur_index - int(selected_EOD[row - return_days][0]) == \
                            return_days:
                        features[cur_index - pad_begin][-1] = \
                            selected_EOD[row][4] / price_max#这里是拿股价除最大股价得到的序列，可以理解为就是原始股价

            except:
                wrong_index.append(stock_index)
                features = np.ones([len(trading_dates) - pad_begin, 6],
                                   dtype=float) * -1234

            #features是最后存的数据
            # # write out
            np.savetxt(os.path.join(opath, self.tickers[stock_index] + '_' + str(return_days) + '.csv'), features, fmt='%.6f', delimiter=',')
        wrong_index = np.array(wrong_index)
        np.save('wrong_index.npy', wrong_index)

if __name__ == '__main__':
    path = '../data'
    
    processor = EOD_Preprocessor(path)
    processor.generate_feature(
        datetime.strptime('2018-01-01', processor.date_format),#这个2013-01-01是数据起始时间不是终点时间
        '../data/stock_data', return_days=1,
        pad_begin=29
    )

