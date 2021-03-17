from DataManager import *
import torch

class Environment: 
    def __init__(self, stock1, stock2, price_col, nrm=1, mode='train'):
        self.stock1 = stock1
        self.stock2 = stock2
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mode = mode
        
        self.price_col = price_col
        
        self.stock1_data, self.stock2_data, self.split_point = load_data("./data/{}.csv", stock1, stock2) # cuda
        self.spread_data, self.stock1_spread, self.stock2_spread, self.coefficient \
            = spread(self.stock1_data[self.price_col], self.stock2_data[self.price_col]) # cuda
        self.spread_mean = self.spread_data.mean()
        self.input_feature = input_feature(self.spread_data) # cuda
        
        self.input_feature = torch.FloatTensor(self.input_feature.to_numpy()).to(self.device)
        
        self.train_stock1_data, self.train_stock2_data = self.stock1_data[:self.split_point], self.stock2_data[:self.split_point].reset_index()
        self.test_stock1_data, self.test_stock2_data = self.stock1_data[self.split_point:], self.stock2_data[self.split_point:].reset_index()
        self.train_spread_data, self.test_spread_data = self.spread_data[:self.split_point], self.spread_data[self.split_point:].reset_index()
        self.train_input_feature, self.test_input_feature = self.input_feature[:self.split_point], self.input_feature[self.split_point:]
        
        if self.mode == 'train':
            print(f"Spread data length : {len(self.train_spread_data)}, input feature length : {len(self.train_input_feature)}")
        elif self.mode == 'test':
            print(f"Spread data length : {len(self.test_spread_data)}, input feature length : {len(self.test_input_feature)}")
        
        self.obv_spread = None
        self.obv_stock1_price = None
        self.obv_stock2_price = None
        self.obv_stock1_spread = None
        self.obv_stock2_spread = None
        self.idx = -1
        
        self.nrm = nrm
        
        self.UPPER_THRESHOLD = 1.05
        self.LOWER_THRESHOLD = 0.95
    
    def reset(self): 
        self.obv_stock1_price = None 
        self.obv_stock2_price = None 
        self.obv_stock1_spread = None 
        self.obv_stock2_spread = None
        self.obv_spread = None 
        
        if self.mode == 'train': 
            if self.idx == len(self.train_input_feature) - 1: 
                self.idx = -1
        elif self.mode == 'test':
            if self.idx == len(self.test_input_feature) - 1:
                self.idx = -1
                
    def observe(self): # next state를 생성하는 함수 
        if self.mode == 'train':
            if len(self.train_input_feature) > self.idx + 1: 
                self.idx += 1
                self.obv_spread = self.train_input_feature[self.idx]
                self.obv_stock1_price = self.train_stock1_data.iloc[self.idx]
                self.obv_stock2_price = self.train_stock2_data.iloc[self.idx]
                return self.obv_spread
        elif self.mode == 'test':
            if len(self.test_input_feature) > self.idx + 1: 
                self.idx += 1
                self.obv_spread = self.test_input_feature[self.idx]
                self.obv_stock1_price = self.test_stock1_data.iloc[self.idx]
                self.obv_stock2_price = self.test_stock2_data.iloc[self.idx]
                return self.obv_spread
        return None 
    
    def get_price(self): # 현재 spread에서의 2개의 주가 가격을 반환하는 함수
        if (self.obv_stock1_price is not None) and (self.obv_stock2_price is not None): 
            return self.obv_stock1_price[self.price_col].item(), self.obv_stock2_price[self.price_col].item()
        return None, None
    
    def hedge_ratio(self):
        return self.coefficient
    
    def step(self, action): 
        new_state = self.observe() # New State
        
        if self.mode == 'train':
            reward = action * self.train_input_feature[self.idx - 1][1] # action * spread_return 
            reward = reward * self.nrm if reward < 0 else reward # train일 때만 음의 reward에 대해서 가중치를 곱함
            done = True if self.idx == len(self.train_input_feature) - 1 else False
        elif self.mode == 'test':
            reward = action * self.test_input_feature[self.idx - 1][1]
            done = True if self.idx == len(self.test_input_feature) - 1 else False
        
        return new_state, reward, done