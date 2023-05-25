import random
import numpy as np
import pandas as pd
import os
import torch as torch
from load_data import load_relation_data, load_EOD_data, select_rel_node, select_price_node
from evaluator import evaluate, evaluate1
from model_pyg import get_loss, GATLSTM_multi_temporal
import time
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

data_path = '../data/stock_data'
parameters = {'seq': 40, 'unit': 64, 'alpha': 1e9}#目前仅'seq'有用
epochs = 800#训练次数
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
trade_record = pd.DataFrame(index = trading_dates[441:1241], columns = ['btl', 'btl5', 'btl10'])
pre_top1 = []
pre_top5 = []
pre_top10 = []
loss_list = []
other_loss_list = []
stop_list = []
model_list = []
a_new_list = []

# relation data
rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
rel_encoding, rel_mask = load_relation_data('../preprocess/汇总.npy', output_type)
'''
list_1, list_2, list_3 = select_rel_node(rel_encoding, 1)
list_4 = select_price_node(price_data, 1)
#delete_list = list(set(list_3) | set(list_4))
delete_list = list_3 + list_4 #list_2 + list_4
delete_list = list(set(delete_list))
#delete_list = list_4

eod_data = np.delete(eod_data, delete_list, axis=0)
mask_data = np.delete(mask_data, delete_list, axis=0)
gt_data = np.delete(gt_data, delete_list, axis=0)
price_data = np.delete(price_data, delete_list, axis=0)
rel_encoding = np.delete(rel_encoding, delete_list, axis=0)
rel_encoding = np.delete(rel_encoding, delete_list, axis=1)
rel_mask = np.delete(rel_mask, delete_list, axis=0)
rel_mask = np.delete(rel_mask, delete_list, axis=1)
'''
'''
new_del = 278
eod_data = eod_data[new_del:,:,:]
mask_data = mask_data[new_del:,:]
gt_data = gt_data[new_del:,:]
price_data = price_data[new_del:,:]
rel_encoding = rel_encoding[new_del:,new_del:]
rel_mask = rel_mask[new_del:,new_del:]
'''


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
        prediction,_ = model(data_batch_1)
        _, _, _, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                 batch_size, parameters['alpha'])# 这里是说prediction是预测股票的收盘价，因为他的return是拿prediction和base_price算的

        cur_valid_pred = cur_rr.cpu()#这里只是改变了排列形式
        cur_valid_gt = gt_batch.cpu()
        cur_valid_mask = mask_batch.cpu()

        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)#这个函数算收益率是对的，model直接输出的不是收益率，是归一化的股价预测
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
        returns = returns_df[col]
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

        # 计算交易胜率
        num_win_trades = (returns > 0).sum()
        num_loss_trades = (returns < 0).sum()
        win_rate = num_win_trades / (num_win_trades + num_loss_trades)

        # 计算交易盈亏比
        avg_win_trade = returns[returns > 0].mean()
        avg_loss_trade = returns[returns < 0].mean()
        profit_loss_ratio = -avg_win_trade / avg_loss_trade

        # 打印计算结果
        print(f"策略 {col} 的指标计算结果如下：")
        print(f"年化收益率：{annual_returns:.2f}%")
        print(f"年化波动率：{annual_volatility:.2f}%")
        print(f"最大回撤率：{max_drawdown:.2f}%")
        print(f"交易胜率：{win_rate:.2f}")
        print(f"交易盈亏比：{profit_loss_ratio:.2f}")



for i in range(200, 400, 100):#数据集开始时间
    model = GATLSTM_multi_temporal(
        batch_size,
        hidden_dim=hidden_dim, 
        num_layers=num_layers,
        rel_encoding=rel_encoding,
        rel_mask=rel_mask
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    
    print(i)
    
    batch_offsets = np.arange(start = i, stop = i + valid_index, dtype=int)#这个时间段就相当于训练集, valid_index相当于决定训练集大小
    # train loop
    for epoch in range(0, epochs):
        np.random.shuffle(batch_offsets)#重新排列batch_offsets
        # steps
        #print(epoch)
        
        for j in range(0, valid_index):#这里的作用大概是从batch_offsets取出16个，意思就是一次取16天
            #T1 = time.time()
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(batch_offsets[j])
            )
            #with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()
            prediction,_ = model(data_batch)
            
            cur_loss, cur_reg_loss, cur_rank_loss, test_pref = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                batch_size, parameters['alpha'])
            
            #mse_loss = torch.nn.MSELoss()
            #reg_loss = mse_loss(prediction, gt_batch)
            # update model
            loss = cur_loss
    
            loss.backward()
            optimizer.step()
            #print([cur_loss, cur_reg_loss, cur_rank_loss])
            if (j+1)%400 ==0:
                print(str(epoch)+': '+str(float(loss)))
            
            loss_list.append(float(loss))
            other_loss_list.append(float(cur_reg_loss))
            #T2 = time.time()
            #print('运行时间:%s秒' % ((T2 - T1)))
        if loss<20:
            stop_list.append(epoch)
            model_list.append(model)
            break
        if epoch == epochs-1:
            stop_list.append(epoch)
            model_list.append(model)
            #a_new_list.append(a_new)
    
    # 回测
    btl = []
    btl5 = []
    btl10 = []
    time_range = max(batch_offsets) + range(1, test_index + 1)
    trading_dates_datch = trading_dates[time_range[0] + parameters['seq'] + 1: time_range[-1] + parameters['seq'] + 1 +1]
    
    for n in range(1, test_index + 1):
        val_perf = validate3(n, model)
        btl.append(val_perf['btl5'])
        btl5.append(val_perf['btl10'])
        btl10.append(val_perf['btl20'])
        pre_top1.append(val_perf['pre_top1'])
        pre_top5.append(val_perf['pre_top5'][:5])
        pre_top10.append(val_perf['pre_top10'])
    
    btl_list = []
    for t in btl:
        btl_list.append(float(t.cpu().detach().numpy()))
    btl_series = pd.Series(btl_list)
    btl_result = btl_series.prod()
    btl_win = sum(btl_series>1) / len(btl_series)
    
    btl5_list = []
    for t in btl5:
        btl5_list.append(float(t.cpu().detach().numpy()))
    btl5_series = pd.Series(btl5_list)
    btl5_result = btl5_series.prod()
    btl5_win = sum(btl5_series>1) / len(btl5_series)
    
    btl10_list = []
    for t in btl10:
        btl10_list.append(float(t.cpu().detach().numpy()))
    btl10_series = pd.Series(btl10_list)
    btl10_result = btl10_series.prod()
    btl10_win = sum(btl10_series>1) / len(btl10_series)
    
    print(btl_result, btl5_result, btl10_result)
    print(btl_win, btl5_win, btl10_win)
    
    trade_record.loc[trading_dates_datch, 'btl5'] = list(btl_series)
    trade_record.loc[trading_dates_datch, 'btl10'] = list(btl5_series)
    trade_record.loc[trading_dates_datch, 'btl20'] = list(btl10_series)