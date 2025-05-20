import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
import torch.nn.functional as F
from agent_dir.agent import Agent
import collections

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        inputs=F.relu(self.fc1(inputs))
        return self.fc2(inputs)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer=collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        mysample=random.sample(self.buffer, batch_size)
        '''
        mysample:[(s,a,r,s',done)],然后对它按列分组存入
        '''
        states,actions,rewards,next_states,dones=zip(*mysample)#对mysample里面的按列分组
        states=torch.tensor(states,dtype=torch.float)   #二维张量[batch_size,state_dim]
        actions=torch.tensor(actions,dtype=torch.int64).view(-1,1) #标量，要转化成[batch_size,1]
        rewards=torch.tensor(rewards,dtype=torch.float).view(-1,1)
        next_states=torch.tensor(next_states,dtype=torch.float)
        dones=torch.tensor(dones,dtype=torch.float).view(-1,1)

        return states,actions,rewards,next_states,dones
    def clean(self):
        self.buffer.clear()
    def size(self):
        return len(self.buffer)


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        self.env=env
        self.state_dim=env.observation_space.shape[0]
        self.action_dim=env.action_space.n
        self.hidden_size=args.hidden_size
        self.lr=args.lr
        self.gamma=args.gamma
        self.epsilon=args.epsilon
        self.count=0
        self.target_update=args.target_update
        self.batch_size=args.batch_size
        self.buffer=ReplayBuffer(args.buffer_size)
        self.device=torch.device("cuda" if args.use_cuda else "cpu")
        self.train_net=QNetwork(self.state_dim,self.hidden_size,self.action_dim).to(self.device)
        self.target_net=QNetwork(self.state_dim,self.hidden_size,self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.train_net.state_dict())
        self.optimizer=optim.Adam(self.train_net.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss()
        self.dqn_type=args.dqn_type
        self.num_episodes=args.num_episodes
        self.min_size=args.min_size#经验回放池最小容量
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.999, patience=5, min_lr=1e-5)
        self.test=args.test
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.env.reset(seed=0)
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        state, _ = self.env.reset(seed=11037)  # 使用默认种子或从参数中获取
        return state
    def train(self):
        """
        Implement your training algorithm here
        """
        return_list=[]
        for i in range(self.num_episodes):
            state=self.init_game_setting()
            episode_return=0
            done=False
            while not done:
                action=self.make_action(state,self.test)
                next_state,reward,terminated,trancated,_=self.env.step(action)
                done=terminated or trancated
                self.buffer.push(state,action,reward,next_state,done)
                state=next_state
                episode_return+=reward
                if self.buffer.size()>=self.min_size:
                    states,actions,rewards,next_states,dones=self.buffer.sample(self.batch_size)
                    self.run(states,actions,rewards,next_states,dones)
            return_list.append(episode_return)
            self.scheduler.step(episode_return)
            print(f'Episode {i}: Return {episode_return},lr {self.scheduler.get_last_lr()}')
        return return_list
    def make_action(self, observation, test):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        action=np.random.randint(self.action_dim)
        if test:
            state=torch.tensor([observation],dtype=torch.float).to(self.device)#[batch_size,state_dim]
            action=self.train_net(state).max(1)[1].item()
        else:
            if np.random.random()<self.epsilon:
                return action
            else:
                state=torch.tensor([observation],dtype=torch.float).to(self.device)
                action=self.train_net(state).max(1)[1].item()
        return action
    def run(self,states,actions,rewards,next_states,dones):
        """
        Implement the interaction between agent and environment here
        """
        states=states.to(self.device)
        actions=actions.to(self.device)
        rewards=rewards.to(self.device)
        next_states=next_states.to(self.device)
        dones=dones.to(self.device)

        q_values=self.train_net(states).gather(1,actions)
        max_next_q=0
        if self.dqn_type=='DoubleDQN':
            best_actions=self.train_net(next_states).argmax(dim=1).view(-1,1)
            max_next_q=self.target_net(next_states).gather(1,best_actions)
        else:
            max_next_q=self.target_net(next_states).max(1)[0].view(-1,1)
        TD_targets=rewards+self.gamma*max_next_q*(1-dones)

        loss=F.mse_loss(q_values,TD_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count+=1
        if self.count%self.target_update==0:
            self.target_net.load_state_dict(self.train_net.state_dict())
