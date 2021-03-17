import torch
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np

def load_data(baseurl, stock1, stock2):
    data1 = pd.read_csv(baseurl.format(stock1), sep=',', encoding='CP949')
    data2 = pd.read_csv(baseurl.format(stock2), sep=',', encoding='CP949')
    
    del data1['Change']
    del data2['Change']
    
    data = data1.merge(data2, on=['Date'], how='left')
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    
    data = data[data['Date'] >= '2014-01-01'].reset_index()
    
    split_point = len(data[data['Date'] <= '2017-12-31'])
    
    stock_data1 = data[['Date', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Volume_x']]
    stock_data1.rename(columns = {'Open_x' : 'Open', "High_x" : 'High', "Low_x" : 'Low', "Close_x" : "Close", 'Volume_x' : 'Volume'}, inplace=True)
    stock_data2 = data[['Date', 'Open_y', 'High_y', 'Low_y', 'Close_y', 'Volume_y']] 
    stock_data2.rename(columns = {'Open_y' : 'Open', "High_y" : 'High', "Low_y" : 'Low', "Close_y" : "Close", 'Volume_y' : 'Volume'}, inplace=True)
 
    return stock_data1, stock_data2, split_point
    
def spread(stock1_data, stock2_data):
    # spread = stock1의 정규화 가격지수 - stock2의 정규화 가격지수
    stock1_normalize = (stock1_data - stock1_data.mean()) / stock1_data.std()
    stock2_normalize = (stock2_data - stock2_data.mean()) / stock2_data.std()
    
    n = round(np.corrcoef(stock1_data, stock2_data)[0, 1], 1)
    #n = coint(stock1_data, stock2_data)[0] / np.var(stock2_data)
    spread = stock1_normalize - stock2_normalize
    #spread = torch.log(stock1_data) - n * torch.log(stock2_data)
    return spread, stock1_normalize, stock2_normalize, n

def input_feature(spread): 
    data = pd.DataFrame(index=range(0, len(spread)), columns=['spread'])
    data['spread'] = spread
    
    data['spread_return'] = spread - spread.shift(1)
    data.loc[0, 'spread_return'] = spread.iloc[0]
    data['MA15'] = spread.rolling(window=15).mean()
    data['MA10'] = spread.rolling(window=10).mean()
    data['MA7'] = spread.rolling(window=7).mean()
    data['MA5'] = spread.rolling(window=5).mean()
    
    for index, i in enumerate(spread[:5]): 
        if index == 0: 
            data.loc[0, 'MA5'] = 1
        else: 
            data.loc[index, 'MA5'] = spread[0:index].mean()
            
    for index, i in enumerate(spread[:7]): 
        if index == 0: 
            data.loc[0, 'MA7'] = 1
        else: 
            data.loc[index, 'MA7'] = spread[0:index].mean()
    
    for index, i in enumerate(spread[:10]): 
        if index == 0: 
            data.loc[0, 'MA10'] = 1
        else: 
            data.loc[index, 'MA10'] = spread[0:index].mean()
    
    for index, i in enumerate(spread[:15]): 
        if index == 0: 
            data.loc[0, 'MA15'] = 1
        else: 
            data.loc[index, 'MA15'] = spread[0:index].mean()
    
    data['MA15/mean'] = spread / data['MA15']
    data['MA10/mean'] = spread / data['MA10']
    data['MA7/mean'] = spread / data['MA7']
    data['MA5/mean'] = spread / data['MA5']
    
    data = data[['spread', 'spread_return', 'MA15', 'MA15/mean', 'MA10', 'MA10/mean', 'MA7', 'MA7/mean', 'MA5', 'MA5/mean']]
    return data

    
    