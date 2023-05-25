import random
import numpy as np
import pandas as pd
import os
import torch as torch
from load_data import load_relation_data, load_EOD_data, select_rel_node, select_price_node
from evaluator import evaluate, evaluate1, evaluate_group
from model_pyg import *
import time
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

model_name = 'GAT_multi_pro'

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

data_path = '../data/stock_data'
parameters = {'seq': 40, 'unit': 64, 'alpha': 1e9}#目前仅'seq'有用
epochs = 60#训练次数
valid_index = 400#测试集长度
test_index = 100#训练集长度
steps = 1
lr=5e-5#学习率

# LSTM参数
hidden_dim = 50  # 隐藏层的神经元个数 50
num_layers = 3   # 隐藏层的层数

# 读取股票数据
tickers_full = pd.read_excel('../data/公司名单_供应.xlsx', index_col=None, sheet_name = 0)
tickers = tickers_full['代码'].values

# 股票、图数据使用范围
output_type = -1#使用的关系数据，-1表示全部，-2表示不使用关系数据

# 读取序列数据
eod_data = np.load("../data/eod_data.npy", allow_pickle=True)
mask_data = np.load("../data/mask_data.npy", allow_pickle=True)
gt_data = np.load("../data/gt_data.npy", allow_pickle=True)
price_data = np.load("../data/price_data.npy", allow_pickle=True)


trade_dates = mask_data.shape[1]
trading_dates = pd.read_excel('../data/trade_day.xlsx')
trading_dates = trading_dates['日期']
trading_dates = trading_dates.iloc[29:]
trade_record = pd.DataFrame(index = trading_dates[441:1241], columns = ['btl10'])


# relation data
rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
rel_encoding, rel_mask = load_relation_data('../preprocess/汇总.npy', output_type)


batch_size = eod_data.shape[0]
full_size = len(tickers)
print(batch_size)


def validate3(n, model):
    with torch.no_grad():
        cur_valid_pred = np.zeros([len(tickers), 1], dtype=float)
        cur_valid_gt = np.zeros([len(tickers), 1], dtype=float)
        cur_valid_mask = np.zeros([len(tickers), 1], dtype=float)

        data_batch_1, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(max(batch_offsets) + n)
        )
        
        try:
            prediction, _ = model(data_batch_1)
        except:
            prediction = model(data_batch_1)
            
        _, _, _, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                 batch_size, parameters['alpha'])# 这里是说prediction是预测股票的收盘价，因为他的return是拿prediction和base_price算的

        cur_valid_pred = cur_rr.cpu()#这里只是改变了排列形式
        cur_valid_gt = gt_batch.cpu()
        cur_valid_mask = mask_batch.cpu()

        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)#这个函数算收益率是对的，model直接输出的不是收益率，是归一化的股价预测
    return cur_valid_perf

def validate_group(n, model):
    """
    get loss on validate/test set
    """
    with torch.no_grad():
        cur_valid_pred = np.zeros([len(tickers), 1], dtype=float)
        cur_valid_gt = np.zeros([len(tickers), 1], dtype=float)
        cur_valid_mask = np.zeros([len(tickers), 1], dtype=float)

        data_batch_1, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(max(batch_offsets) + n)
        )
        
        try:
            prediction, _ = model(data_batch_1)
        except:
            prediction = model(data_batch_1)
            
        _, _, _, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                 batch_size, parameters['alpha'])# 这里是说prediction是预测股票的收盘价，因为他的return是拿prediction和base_price算的
        cur_valid_pred = cur_rr.cpu()#这里只是改变了排列形式
        cur_valid_gt = gt_batch.cpu()
        cur_valid_mask = mask_batch.cpu()

        cur_valid_perf = evaluate_group(cur_valid_pred, cur_valid_gt, cur_valid_mask)#这个函数算收益率是对的，model直接输出的不是收益率，是归一化的股价预测
    return cur_valid_perf

def get_batch(offset):
    seq_len = parameters['seq']
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],#855到871
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

def calculate_metrics(returns_df):
    for col in returns_df.columns:
        returns = returns_df[col] - 1
        # 计算年化收益率
        annual_returns = ((1 + returns.mean()) ** 252 - 1) * 100

        # 计算年化波动率
        annual_volatility = returns.std() * np.sqrt(252) * 100

        # 计算最大回撤率
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = 0
        for t in range(1, len(cumulative_returns)):
            peak = cumulative_returns.iloc[:t].max()
            drawdown = (cumulative_returns.iloc[t] - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        max_drawdown = max_drawdown * 100
        
        # 计算交易胜率
        num_win_trades = (returns > 0).sum()
        num_loss_trades = (returns < 0).sum()
        win_rate = num_win_trades / (num_win_trades + num_loss_trades)*100

        # 计算交易盈亏比
        avg_win_trade = returns[returns > 0].mean()
        avg_loss_trade = returns[returns < 0].mean()
        profit_loss_ratio = -avg_win_trade / avg_loss_trade

        # 计算卡玛比率
        kama_ratio = -cumulative_returns[-1] * 100 / (max_drawdown + 1)
        
        # 计算夏普比率
        risk_free_rate = 0.02 # 假设无风险利率为2%
        sharpe_ratio = (annual_returns - risk_free_rate) / annual_volatility

        cumulative_returns = (cumulative_returns[-1]-1) * 100
        # 打印计算结果
        print(f"策略 {col} 的指标计算结果如下：")
        print(f"累计收益率：{cumulative_returns:.2f}%")
        print(f"年化收益率：{annual_returns:.2f}%")
        print(f"最大回撤率：{max_drawdown:.2f}%")
        print(f"交易胜率：{win_rate:.2f}")
        print(f"交易盈亏比：{profit_loss_ratio:.2f}")
        print(f"夏普比率：{sharpe_ratio:.2f}\n")

preformance_group_list = []


for i in range(0, 800, 100):#数据集开始时间
    print('\n阶段' + str(int((i+1)/100)) + '结果')
    model = torch.load(model_name + str(int(i/100)) + '.pt')
    
    batch_offsets = np.arange(start = i, stop = i + valid_index, dtype=int)#这个时间段就相当于训练集, valid_index相当于决定训练集大小
    
    # 回测
    btl10 = []
    time_range = max(batch_offsets) + range(1, test_index + 1)
    trading_dates_datch = trading_dates[time_range[0] + parameters['seq'] + 1: time_range[-1] + parameters['seq'] + 1 +1]
    
    for n in range(1, test_index + 1):
        val_perf = validate3(n, model)
        btl10.append(val_perf['btl10'])
    
    btl10_list = []
    for t in btl10:
        btl10_list.append(float(t.cpu().detach().numpy()))
    btl10_series = pd.Series(btl10_list)
    btl10_result = btl10_series.prod()
    btl10_win = sum(btl10_series>1) / len(btl10_series)
    
    trade_record.loc[trading_dates_datch, 'btl10'] = list(btl10_series)
    
    calculate_metrics(trade_record.loc[trading_dates_datch, :])
    '''
    for n in range(1, test_index + 1):
        preformance_group = validate_group(n, model)
        preformance_group_list.append(preformance_group)
    '''

'''
group_record = pd.DataFrame(preformance_group_list, index = trading_dates[441:1241], columns = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6', 'Group7', 'Group8', 'Group9', 'Group10'])
cumulative_returns = (1 + group_record).cumprod()

ax = cumulative_returns.plot()
ax.legend(fontsize=8)
ax.set_ylabel('Net Worth')
plt.show()
#plt.savefig("分层图_"+model_name, dpi=300)
'''

'''
trade_record.columns = ['Net Worth']
cumulative_returns = trade_record.cumprod()
ax = cumulative_returns.plot()
ax.legend(fontsize=8)
ax.set_ylabel('Net Worth')
plt.show()
#plt.savefig("净值_"+model_name, dpi=300)
'''