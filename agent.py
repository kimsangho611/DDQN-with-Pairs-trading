from ReplayMemory import *
import networks
import numpy as np
import torch
import torch.nn.functional as F

class Agent:
    TRADING_CHANGE = 0.00015 # 거래 수수료 (0.015%)
    TRADING_TAX = 0.0025 # 거래세 (0.25%)
    
    # 학습시 입력 수
    AGENT_STATE = 10
    # 행동
    ACTION_LONG = 1
    ACTION_SHORT = -1
    ACTION_NO_POSITION = 0
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_LONG, ACTION_NO_POSITION, ACTION_SHORT]
    NUM_ACTIONS = len(ACTIONS)
    
    def __init__(self, environment, state_dim, action_dim, capacity, batch_size, lr=0.01, gamma=0.9, mode='train'): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU로 돌리기 위한 변수
        self.mode = mode
        self.gamma = gamma
        
        self.environment = environment
        
        self.capacity = capacity
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.target_q_network = networks.DNN(state_dim).to(self.device)
        self.evaluation_q_network = networks.DNN(state_dim).to(self.device)
        if self.mode == 'test':
            self.target_q_network.load_state_dict(torch.load("./model_save/{}{}_{}_{}.pth".format("t", self.environment.stock1, self.environment.stock2,
                                                                                                  int(self.environment.nrm))))
            self.evaluation_q_network.load_state_dict(torch.load("./model_save/{}{}_{}_{}.pth".format('e', self.environment.stock1, self.environment.stock2,
                                                                                                      int(self.environment.nrm))))
        
        self.optimizer = torch.optim.Adam(self.target_q_network.parameters(), lr=lr)
    
        self.state_action_value = None
        self.expected_state_action_values = None
    
    def get_action(self, state): 
        self.target_q_network.eval()
        with torch.no_grad(): 
            action = self.target_q_network(state.unsqueeze(0)).max(1)[1]
        return action # action의 경우 -1, 0, 1로 구성되어 있음 (0 ==> short, 1 ==> no position, 2 ==> long)
            
    
    def update_q_function(self): # train target_q_network using replay memory
        if len(self.memory) < self.batch_size: 
            return 
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.next_state_batch = \
            self.make_minibatch() # batch를 생성하는 부분
            
        self.expected_state_action_values = self.get_expected_state_action_value() # batch를 학습하는 부분
        
        self.update_target_q_network() # 학습한 부분을 network에 업데이트하는 부분
    
    def memorize(self, state, action, state_next, reward): 
        self.memory.push(state, action, state_next, reward) # 경험을 저장
    
    def update_evaluation_q_function(self): # target_q_network를 evaluation_q_network에 덮어씀
        self.evaluation_q_network.load_state_dict(self.target_q_network.state_dict())
        
    def update_target_q_network(self):
        self.target_q_network.train()
        
        loss = F.mse_loss(self.state_action_value, self.expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def make_minibatch(self): 
        transition = self.memory.sample(self.batch_size)
        
        batch = self.memory.Transition(*zip(*transition))
        
        state_batch = torch.cat(batch.state).reshape(-1, self.state_dim)
        action_batch = torch.cat(batch.action).reshape(-1, 1)
        reward_batch = torch.cat(batch.reward).reshape(-1, 1)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).reshape(-1, self.state_dim)
        
        return batch, state_batch, action_batch, reward_batch, next_state_batch
    
    def get_expected_state_action_value(self): 
        self.target_q_network.eval()
        self.evaluation_q_network.eval()

        self.state_action_value = self.target_q_network(self.state_batch).gather(1, self.action_batch)
        
        final_mask = torch.cuda.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        next_state_value = torch.zeros(self.batch_size).to(self.device)
        a_m = torch.zeros(self.batch_size).type(torch.cuda.LongTensor)
        
        a_m[final_mask] = self.target_q_network(self.next_state_batch).detach().max(1)[1]
        a_m_final_next_states = a_m[final_mask].view(-1, 1)
        
        next_state_value[final_mask] = self.evaluation_q_network(self.next_state_batch).gather(1, a_m_final_next_states).detach().squeeze()
        
        return self.reward_batch + self.gamma * next_state_value