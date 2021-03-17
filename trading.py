from agent import Agent
from environment import Environment
from StockInfo import StockInfo
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Trading: 
    def __init__(self, stock1, stock2, initial_balance=10000, capacity=200, batch_size=100, nrm=1, episode=300, gamma=0.9, lr=0.01, mode='train'): 
        self.stock1 = stock1
        self.stock2 = stock2
        self.mode = mode

        self.episodes = episode
        self.nrm = nrm
        
        self.environment = Environment(stock1, stock2, 'Close', self.nrm, mode=mode)
        self.agent = Agent(self.environment, state_dim=10, action_dim=3, capacity=capacity, batch_size=batch_size, lr=lr, gamma=gamma, mode=mode)
        self.stockinfo = StockInfo(self.environment, initial_balance)
        
    def run(self): 
        for episode in range(self.episodes): # 300
            print("Current Episode : ", episode + 1)
            self.environment.reset() # 초기 상태로 reset
            state = self.environment.observe()
            
            while True: 
                action = self.agent.get_action(state) # 0인 경우 short, 1인 경우 no position, 2인 경우 longs
                
                # 현재 2개의 주가 
                stock1_price, stock2_price = self.environment.get_price()
                self.stockinfo.trade(action - 1)
                next_state, reward, done = self.environment.step(action - 1)
                
                self.agent.memorize(state, action, next_state, reward) # capacity: 200, batch_size: 100
                self.agent.update_q_function()
                
                state = next_state
                
                if done: 
                    break
            self.agent.update_evaluation_q_function()
            
            self.stockinfo.print_info()
            self.stockinfo.reset()
            print()
            
        if self.mode == 'train':
            torch.save(self.agent.target_q_network.state_dict(), "./model_save/{}{}_{}_{}.pth".format("t", self.stock1, self.stock2, int(self.nrm)))
            torch.save(self.agent.evaluation_q_network.state_dict(), "./model_save/{}{}_{}_{}.pth".format("e", self.stock1, self.stock2, int(self.nrm)))
            
            with open("./log/tn_{}_{}_{}.csv".format(self.stock1, self.stock2, int(self.nrm)), 'w') as f:
                f.write("{},{},{},{},{},{},{}\n".format("long", "no position", "short", "profit", "Stock1 Trade Count", "Stock2 Trade Count", "Both Trade Count"))
                for i in range(self.episodes):
                    f.write("{},{},{},{},{},{},{}\n".format(
                        self.stockinfo.long_log[i], self.stockinfo.no_position_log[i], self.stockinfo.short_log[i], self.stockinfo.PV_log[i],
                        self.stockinfo.stock1_trade_log[i], self.stockinfo.stock2_trade_log[i], self.stockinfo.both_trade_log[i]))
        elif self.mode == 'test':
            with open("./log/tt_{}_{}_{}.csv".format(self.stock1, self.stock2, int(self.nrm)), 'w') as f:
                f.write("{},{},{},{},{},{},{}\n".format("long", "no position", "short", "profit", "Stock1 Trade Count", "Stock2 Trade Count", "Both Trade Count"))
                f.write("{},{},{},{},{},{},{}\n".format(
                        self.stockinfo.long_log[0], self.stockinfo.no_position_log[0], self.stockinfo.short_log[0], self.stockinfo.PV_log[0],
                        self.stockinfo.stock1_trade_log[0], self.stockinfo.stock2_trade_log[0], self.stockinfo.both_trade_log[0]))
                
    
if __name__=='__main__': 
    stock_pairs = [('BEN', 'COG'), ('DISCA', 'RIG'), ("DISCK", 'RIG'), ("ADBE", 'CRM'), ("CF", "HBI"), ("ESV", "GNW"), ("CNX", "HBI"), ("AMZN", "CRM"), 
             ("MA", "VFC"), ("FCX", "GNW"), ("CRM", "NVDA"), ("CF", "FOSL"), ("FCX", "HBI"), ("DISCK", "ESV"), ("DISCA", "ESV"), ("ESV", "RRC"), ("NBL", "RIG"), 
             ("CNX", "GNW"), ("COG", "DO"), ("HBI", "NBL"), ("HBI", "MRO"), ("GNW", "NBL"), ("DISCA", "MA"), ("DISCK", "MA"), ("RIG", "RRC"), ("CF", "CNX"),
             ("CF", "GNW"), ("ESV", "HBI"), ("ADBE", "RHT"), ("MA", "RIG"), ("NBL", "SWN"), ("AWR", "WTR"), ("SLB", "PFE")]
    nrm = [1, 2.5, 5, 10, 20, 50, 100, 200, 500, 700, 1000]
    
    trading = Trading("COG", "DO", episode=300, nrm=5, mode='train')
    trading.run()
    """for index, stock in tqdm(enumerate(stock_pairs)):
        for n in nrm: 
            trading = Trading(stock[0], stock[1], episode=300, nrm=n, mode='train')
            trading.run()
        
            trading = Trading(stock[0], stock[1], episode=1, nrm=1, mode='test')
            trading.run()"""