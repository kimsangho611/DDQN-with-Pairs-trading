import numpy as np

class StockInfo:
    def __init__(self, environment, initial_balance):
        self.environment = environment
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        self.stock1_unit = 0
        self.stock1_blank_unit = 0
        self.stock1_avg_price = 0
        self.stock1_hold_wait2sell = False
        self.stock2_unit = 0
        self.stock2_blank_unit = 0
        self.stock2_avg_price = 0
        self.stock2_hold_wait2sell = False 
        
        self.PV = 0
        self.base_PV = 0
        
        self.num_long = 0
        self.num_short = 0
        self.num_no_position = 0
        self.num_trading1 = 0
        self.num_trading2 = 0
        self.num_both_trading = 0
        self.is_trade1 = False
        self.is_trade2 = False
        
        self.long_log = []
        self.no_position_log = []
        self.short_log = []
        self.PV_log = []
        self.stock1_trade_log = []
        self.stock2_trade_log = []
        self.both_trade_log = []
        
        self.profitloss = 0
        
        self.TRADING_TAX = 0.0025
        self.UPPER_THRESHOLD = 1.05
        self.LOWER_THRESHOLD = 0.95
        
        self.hedge_ratio = self.environment.hedge_ratio()
        self.stock1_trade_unit = 10
        self.stock2_trade_unit = 10 * self.hedge_ratio
        
        self.num = 1
    
    def reset(self):
        self.num_long = 0
        self.num_no_position = 0
        self.num_short = 0
        self.num_trading1 = 0
        self.num_trading2 = 0
        self.num_both_trading = 0
        self.is_trade1 = False
        self.is_trade2 = False
        
        self.PV = 0
        self.base_PV = 0
        
        self.balance = self.initial_balance
        self.profitloss = 0
        
        self.stock1_unit = 0
        self.stock1_blank_unit = 0
        self.stock1_avg_price = 0
        self.stock1_hold_wait2sell = False
        self.stock2_unit = 0
        self.stock2_blank_unit = 0
        self.stock2_avg_price = 0
        self.stock2_hold_wait2sell = False 
        
        
    def trade(self, action):
        self.is_trade1 = False
        self.is_trade2 = False
        stock1_price, stock2_price = self.environment.get_price()
        if stock1_price is None or stock2_price is None:
            return
        
        if action == -1: # Short signal == > stock1 sell & stock2 buy
            self.num_short += 1
            # stock1 trade ---------------------------------
            if self.stock1_unit > 0: 
                trade = self.num
                if self.stock1_unit - trade >= 0:
                    invest_mount = stock1_price * trade * (1 - self.TRADING_TAX)
                    if invest_mount > 0: 
                        self.stock1_unit -= trade
                        self.balance += invest_mount
                        self.is_trade1 = True
                        
                else: # 공매도를 시행해야함 (일단 공매도 말고 있는 주식을 모두 다 파는 걸로)
                    invest_mount = stock1_price * self.stock1_unit * (1 - self.TRADING_TAX)
                    if invest_mount > 0:
                        self.balance += invest_mount
                        self.stock1_unit = 0  
                        self.is_trade1 = True              
            # stock2 trade ---------------------------------
            trade = self.num
            invest_mount = stock2_price * trade * (1 - self.TRADING_TAX)
            if self.balance > invest_mount:
                self.balance -= invest_mount
                self.stock2_unit += trade 
                self.is_trade2 = True
                
            
        elif action == 0: # No position signal ==> 모든 포트폴리오를 닫음
            self.num_no_position += 1
            
            if self.stock1_unit > 0:
                invest_mount = stock1_price * self.stock1_unit * (1 - self.TRADING_TAX)
                self.balance += invest_mount
                self.stock1_unit = 0 
                self.is_trade1 = True          
            if self.stock2_unit > 0:
                invest_mount = stock2_price * self.stock2_unit * (1 - self.TRADING_TAX)
                self.balance += invest_mount
                self.stock2_unit = 0
                self.is_trade2 = 0
                
        elif action == 1: # Long signal == > stock1 buy & stock2 sell
            self.num_long += 1
            # stock1 buy ------------------------------------
            trade = self.num
            invest_mount = stock1_price * trade * (1 - self.TRADING_TAX)
            if self.balance > invest_mount:
                self.balance -= invest_mount
                self.stock1_unit += trade
                self.is_trade1 = True
            # stock2 sell -----------------------------------
            if self.stock2_unit > 0:
                trade = self.num
                if self.stock2_unit - trade >= 0:
                    invest_mount = stock2_price * trade * (1 - self.TRADING_TAX)
                    if invest_mount > 0:
                        self.balance += invest_mount
                        self.stock2_unit -= trade
                        self.is_trade2 = True
                else:
                    invest_mount = stock2_price * self.stock2_unit * (1 - self.TRADING_TAX)
                    if invest_mount > 0:
                        self.balance += invest_mount
                        self.stock2_unit = 0
                        self.is_trade2 = True
                        
        if self.is_trade1 and self.is_trade2:
            self.num_both_trading += 1
        else:  
            if self.is_trade1:
                self.num_trading1 += 1
            if self.is_trade2:
                self.num_trading2 += 1
    
        self.PV = self.balance + (stock1_price * self.stock1_unit) + (stock2_price * self.stock2_unit)
        self.profitloss = (self.PV - self.initial_balance) * 100 / self.initial_balance
    
    def get_pattern(self, spread, spread_mean):
        pass 
            
    
    def CalUnitPrice(self, stock_hold_unit, stock_avg_price, stock_new_unit, stock_new_price):
        avg_price = ((stock_hold_unit * stock_avg_price) + (stock_new_unit * stock_new_price)) \
            /(stock_hold_unit + stock_new_unit)
        return avg_price
    
    def print_info(self):
        print("Stock1 : {}    Stock2 : {}     nrm : {}".format(self.environment.stock1, self.environment.stock2, self.environment.nrm))
        print("{} unit : ".format(self.environment.stock1), self.stock1_unit, "    {} unit : ".format(self.environment.stock2), self.stock2_unit)
        print(f"Stock1 Trading Count : {self.num_trading1}   Stock2 Trading Count : {self.num_trading2}  Both Trading Count : {self.num_both_trading}")
        print("Portfolio Value : {:.3f}  Profit Loss: {:.4f}%".format(self.PV, self.profitloss))
        print(f"Long position : {self.num_long}   No position : {self.num_no_position}   Short position : {self.num_short}")
        self.long_log.append(self.num_long)
        self.no_position_log.append(self.num_no_position)
        self.short_log.append(self.num_short)
        self.PV_log.append(self.profitloss)
        self.stock1_trade_log.append(self.num_trading1)
        self.stock2_trade_log.append(self.num_trading2)
        self.both_trade_log.append(self.num_both_trading)