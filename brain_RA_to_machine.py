import random
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path  # call this otherwise it will raise excepetion if you use another computer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import routing

from tabulate import tabulate
import logging
import os

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='brain_training_RA.log',  # 将日志输出到文件
    filemode='w'  # 覆盖模式（'w'），每次运行清空文件；追加模式（'a'）则保留历史日志
)

class routing_brain:
    def __init__(self, env, job_creator, m_list, wc_list, warm_up, span, *args, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.job_creator = job_creator
        self.m_list = m_list
        self.wc_list = wc_list
        # specify the constituent machines, and activate their routing learning
        self.m_per_wc = len(self.wc_list[0].m_list)
        print(self.wc_list[0].m_list,self.m_per_wc)
        for m in m_list:
            m.routing_learning_event.succeed()
        # specify the path to store the model
        self.path = sys.path[0]
        # state space, eah machine generate 3 types of data, along with the processing time of arriving job, and job slack
        self.input_size = 17
        # action space, consists of all selectable machines
        for wc in self.wc_list:
            wc.job_routing = self.action_warm_up
        self.func_list = [routing.TT, routing.ET, routing.EA, routing.SQ]
        self.func_selection = 0
        self.output_size = len(self.func_list)
        # learning rate
        self.lr = 0.01

        self.log_info = 0
        # specify the address to store the model

        if 'TEST' in kwargs and kwargs['TEST']:
            print("---X TEST mode ON X---")
            self.address_seed = "{}\\routing_models\\TEST_state_dict"
            self.routing_action_NN = bulid_network_validated(self.input_size, self.output_size)
            self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
            self.build_state = self.state_mutil_channel
            self.train = self.train_DDQN
            for wc in self.wc_list:
                wc.build_state = self.state_mutil_channel


        ''' specify the optimizer '''
        self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr=self.lr, momentum = 0.9)
        # Initialize the parameters for training
        self.minibatch_size = 128
        self.rep_memo_size = 1024
        self.discount_factor = 0.8 # how much agent care long-term rewards
        self.epsilon = 0.5  # exploration
        # parameters of training
        self.warm_up = warm_up
        self.span = span
        self.routing_action_NN_training_interval = 2
        self.routing_action_NN_training_time_record = []
        self.routing_target_NN_sync_interval = 250  # synchronize the weights of NN every 500 time units
        self.routing_target_NN_update_time_record = []
        # initialize the list for deta memory
        self.rep_memo = []
        self.data_memory = []
        self.exploration_record = []
        self.time_record = []
        self.loss_record = []
        self.delete_obsolete_experience = self.env.event()
        # processes
        if 'global_reward' in kwargs and kwargs['global_reward']:
            self.address_seed += "_globalR"
            for m in self.m_list:
                m.routing_global_reward = True
            for wc in self.wc_list:
                wc.build_routing_experience = wc.complete_experience_global_reward
        if 'IQL' in kwargs and kwargs['IQL'] or 'I_DDQN' in kwargs and kwargs['I_DDQN'] :
            self.env.process(self.training_process_independent())
            self.env.process(self.update_rep_memo_independent_process())
            self.rep_memo = {} # replace the list by dict
            for wc in self.wc_list:
                self.rep_memo[wc.wc_idx] = []
            self.build_initial_rep_memo = self.build_initial_rep_memo_independent
            self.env.process(self.update_training_parameters_process())
            #self.rep_memo_size /= self.m_no # size for independent replay memory
        else: # default mode is parameter sharing
            self.env.process(self.training_process_parameter_sharing())
            self.env.process(self.update_rep_memo_parameter_sharing_process())
            self.build_initial_rep_memo = self.build_initial_rep_memo_parameter_sharing
            self.env.process(self.update_learning_rate_process())
        # finalize the address_seed
        self.address_seed += ".pt"
        # some shared process
        self.env.process(self.warm_up_process())
        self.env.process(self.update_training_setting_process())


    '''
    1. downwards for functions that required for the simulation
       including the warm-up, action functions and multiple sequencing rules
    '''


    def warm_up_process(self):
        '''
        Phase 1.1 : warm-up
        within this phase, agent use default routing rule
        '''
        # take over the target workcenter's routing function
        # from routing.earliest_available to here
        # 一开始的时候，就让工作中心的路由函数变成 random exploration
        print('+++ Take over the routing function of target workcenter +++')
        # for wc in self.wc_list:
        #     wc.job_routing = self.CT
        #     wc.job_routing = self.action_random_exploration
        # # upper half of warm-up period
        # yield self.env.timeout(0.9*self.warm_up)
        for idx,func in enumerate(self.func_list):
            self.func_selection = idx
            print('set to rule {}'.format(func))
            for wc in self.wc_list:
                self.func_selection = idx
            yield self.env.timeout(self.warm_up/10)
        
        print("Warm-up period accomplished, start random exploration", self.env.now)
        '''
        Phase 1.2 : random exploration
        within this phase, agent make random routing action, to accumulate experience
        '''
        # change the target workcenter's routing function to random exploration
        for wc in self.wc_list:
            wc.job_routing = self.action_random_exploration
        # lower half of warm-up period
        yield self.env.timeout(0.1*self.warm_up-1)
        print("Random exploration accomplished, start training", self.env.now)
        # after the warm up period, build replay memory and start training
        self.build_initial_rep_memo()
        # hand over the target workcenter's routing function to policy network
        for wc in self.wc_list:
            wc.job_routing = self.action_DRL

    # use as the default, initial routing rule for workcenter
    def EA(self, job_idx, routing_data, job_pt,ttd, job_slack, wc_idx, *args):
        # concatenate all data, build state, dtype is float
        # s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)
        # axis=0 means choose along columns
        rank = np.argmin(routing_data, axis=0)
        a_t = torch.tensor(rank[1])
        # add current state and action to target wc's incomplete experience
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        print('ET ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[a_t].m_idx))
        return a_t

    def CT(self, job_idx, routing_data, job_pt,ttd, job_slack, wc_idx, *args): # earliest completion time
        # s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)

        #print(data,job_pt)
        completion_time = np.array(routing_data)[:,1].clip(0) + np.array(job_pt)
        rank = completion_time.argmin()
        a_t = torch.tensor(rank)
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        # print('CT time ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[a_t].m_idx))
        return a_t

    # used in the first half phase of warm up 每一个func 都来一遍
    def action_warm_up(self, job_idx, routing_data, job_pt,ttd, job_slack, wc_idx, *args):
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)

        # action is index of rule, NOT index or position of job
        a_t = torch.tensor(self.func_selection)
        machine_position = self.func_list[self.func_selection](job_idx ,routing_data, ttd, job_pt, job_slack, wc_idx)
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        return machine_position
    
    # random exploration is for collecting more experience
    # routing_data 代表了 routing 方法 选择哪个机器
    # args[2] 代表了那些需要被处理进网络训练的数据
    def action_random_exploration(self, job_idx, routing_data, job_pt, ttd, job_slack, wc_idx, *args):
        # s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)
        # generate a random number in [0, self.m_per_wc)
        # a_t = torch.randint(0,self.m_per_wc,[])
        self.func_selection = np.random.randint(self.output_size)
        a_t = torch.tensor(self.func_selection)
        machine_position = self.func_list[self.func_selection](job_idx ,routing_data, ttd, job_pt, job_slack, wc_idx)
        # add current state and action to target wc's incomplete experience
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        # logging.info("Job %s machine%s: action: %s initial full %s  time %s", job_idx, machine_position, a_t, s_t, self.env.now)
        # print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        return machine_position

    def action_DRL(self, job_idx, routing_data, job_pt, ttd, job_slack, wc_idx, *args):
        # s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        s_t = self.build_state(routing_data, job_pt, ttd, job_slack, wc_idx)
        # generate the action
        if random.random() < self.epsilon:
            # a_t = torch.randint(0,self.m_per_wc,[])
            a_t = torch.randint(0,len(self.func_list),[])
            machine_position = self.func_list[self.func_selection](job_idx, routing_data, ttd, job_pt, job_slack, wc_idx)
            # print('RANDOM ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        else:
            # input state to policy network, produce the state-action value
            value = self.routing_action_NN.forward(s_t.reshape(1,1,self.input_size), wc_idx)
            # print("State:",s_t, 'State-Action Values:', value)
            a_t = torch.argmax(value)
            machine_position = self.func_list[self.func_selection](job_idx, routing_data, ttd, job_pt, job_slack, wc_idx)
            # print('DRL ROUTING: wc {} assign job {} to m {}'.format(wc_idx, job_idx,self.wc_list[wc_idx].m_list[machine_position].m_idx))
        # add current state and action to target wc's incomplete experience
        # this is the first half of a single record of experience
        # the second half will be appended to first half when routed job complete its operation
        self.build_experience(job_idx, s_t, a_t, wc_idx)
        self.time_record.append(self.env.now)
        return machine_position

    def build_experience(self, job_idx, s_t, a_t, wc_idx):
        self.wc_list[wc_idx].incomplete_experience[job_idx] = [s_t, a_t]

    '''
    2. downwards are functions used for building the state of the experience (replay memory)
    '''

    def state_deeper(self, routing_data, job_pt, ttd, job_slack, wc_idx):
        coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]  # return the index of coming jobs
        coming_job_no = coming_job_idx.size  # expected arriving job number
        if coming_job_no:  # if there're jobs coming at your workcenter
            next_job = self.job_creator.release_time_list[coming_job_idx].argmin()  # the index of next job
            coming_job_time = (self.job_creator.release_time_list[coming_job_idx] - self.env.now)[next_job]  # time from now when next job arrives at workcenter
            coming_job_slack = self.job_creator.arriving_job_slack_list[coming_job_idx][next_job]  # what's the average slack time of the arriving job
        else:
            coming_job_time = 0
            coming_job_slack = 0

        # Ensure all inputs are numpy arrays with consistent dtype  取前两个数
        m_state = np.array([a[:2] for a in routing_data], dtype=np.float32)  # Convert to 2D array
        job_pt = np.array([job_pt], dtype=np.float32)  # Convert to 1D array
        job_slack = np.array([job_slack], dtype=np.float32)  # Convert to 1D array
        coming_job_time = np.array([coming_job_time], dtype=np.float32)  # Convert to 1D array
        coming_job_slack = np.array([coming_job_slack], dtype=np.float32)  # Convert to 1D array
        # print("m_state:", m_state, m_state.shape, m_state.dtype)
        # print("job_pt:", job_pt, job_pt.shape, job_pt.dtype)
        # print("job_slack:", job_slack, job_slack.shape, job_slack.dtype)
        # print("coming_job_time:", coming_job_time, coming_job_time.shape, coming_job_time.dtype)
        # print("coming_job_slack:", coming_job_slack, coming_job_slack.shape, coming_job_slack.dtype)
        # Flatten all arrays
        m_state_flat = m_state.flatten()  # 展平为 (4,)
        job_pt_flat = job_pt.flatten()  # 展平为 (2,)
        job_slack_flat = job_slack.flatten()  # 展平为 (1,)
        coming_job_time_flat = coming_job_time.flatten()  # 展平为 (1,)
        coming_job_slack_flat = coming_job_slack.flatten()  # 展平为 (1,)

        # Concatenate all arrays
        s_t_np = np.concatenate([m_state_flat, job_pt_flat, job_slack_flat, coming_job_time_flat, coming_job_slack_flat])

        # Convert to PyTorch tensor
        s_t = torch.tensor(s_t_np, dtype=torch.float32)        
        # print(s_t, s_t.shape, s_t.dtype)
        return s_t

    def state_deeper_new(self, routing_data, job_pt, ttd, job_slack, wc_idx):
        st = np.concatenate([routing_data])
        s_t = torch.tensor(st, dtype=torch.float32)
        return s_t

    def state_mutil_channel(self, routing_data, job_pt, ttd, job_slack, wc_idx):
        coming_job_idx = np.where(self.job_creator.next_wc_list == wc_idx)[0]  # return the index of coming jobs
        coming_job_no = coming_job_idx.size  # expected arriving job number
        # if coming_job_no:  # if there're jobs coming at your workcenter
        #     next_job = self.job_creator.release_time_list[coming_job_idx].argmin()  # the index of next job
        #     coming_job_time = (self.job_creator.release_time_list[coming_job_idx] - self.env.now)[next_job]  # time from now when next job arrives at workcenter
        #     coming_job_slack = self.job_creator.arriving_job_slack_list[coming_job_idx][next_job]  # what's the average slack time of the arriving job
        # else:
        #     coming_job_time = 0
        #     coming_job_slack = 0
        # update the training data for routing
        in_system_job_no = self.job_creator.in_system_job_no
        buffer_num = sum(sublist[2] for sublist in routing_data)
        # print('buffer_num:', buffer_num, 'routing_data:', routing_data)
        buffer_num_mean = buffer_num / len(routing_data)

        job_pt = np.array([job_pt], dtype=np.float32)  # Convert to 1D array
        job_pt_sum = np.sum(job_pt)
        job_pt_mean = np.mean(job_pt)
        job_pt_min = np.min(job_pt)
        job_pt_std_dev = np.std(job_pt)
        job_pt_cv = job_pt_std_dev / job_pt_mean

        m_available_list = [sublist[1] for sublist in routing_data]
        m_available_sum = sum(m_available_list)
        m_available_min = min(m_available_list)
        m_available_mean = np.mean(m_available_list)
        m_available_std_dev = np.std(m_available_list)
        # 检查 m_available_mean 是否为 0 或 NaN
        if m_available_mean == 0 or np.isnan(m_available_mean):
            m_available_cv = 0  # 或其他默认值
        else:
            m_available_cv = m_available_std_dev / m_available_mean

        # job_slack = np.array([job_slack], dtype=np.float32)  # Convert to 1D array
        # ttd = np.array([ttd], dtype=np.float32)  # Convert to 1D array

        m_ur_list = [sublist[4] for sublist in routing_data]
        # print('m_ur_list:', m_ur_list)
        ur_mean = np.mean(m_ur_list)
        ur_std_dev = np.std(m_ur_list)
        # 计算 ur_cv
        if ur_mean == 0 or np.isnan(ur_mean):
            ur_cv = 0  # 或其他默认值
        else:
            ur_cv = ur_std_dev / ur_mean

        # information of progression of jobs, get from the job creator
        global_comp_rate = self.job_creator.comp_rate
        global_realized_tard_rate = self.job_creator.realized_tard_rate
        global_exp_tard_rate = self.job_creator.exp_tard_rate

        # use raw data, and leave the magnitude adjustment to normalization layers
        no_info = [in_system_job_no, buffer_num_mean, coming_job_no] # info in job number
        pt_info = [job_pt_sum, job_pt_mean, job_pt_min] # info in processing time
        m_available_info = [m_available_sum, m_available_mean, m_available_min] # info in machine available time
        ttd_slack_info = [job_slack, ttd] # info in time till due
        progression = [global_comp_rate, global_realized_tard_rate, global_exp_tard_rate] # progression info
        heterogeneity = [job_pt_cv, m_available_cv, ur_cv] # heterogeneity
        # print('no_info:', no_info, 'pt_info:', pt_info, 'm_available_info:', m_available_info, 'ttd_slack_info:', ttd_slack_info, 'progression:', progression, 'heterogeneity:', heterogeneity)
        # concatenate the data input
        s_t = np.nan_to_num(np.concatenate([no_info, pt_info, m_available_info, ttd_slack_info, progression, heterogeneity]),nan=0,posinf=1,neginf=-1)
        # convert to tensor
        s_t = torch.tensor(s_t, dtype=torch.float)
        return s_t
        

    '''
    3. downwards are functions used for building / updating replay memory
    '''
    def convert_rep_memo(self, rep_memo):
        """
        将 rep_memo 中的每个元素转换为 Python 原生类型（列表或标量）。
        
        参数:
            rep_memo (list): 原始的经验回放池，每个元素是一个元组 (s_t, a_t, r_t, s_t_plus_1)。
            
        返回:
            list: 转换后的经验回放池，每个元素是一个元组 (s_t, a_t, r_t, s_t_plus_1)。
        """
        rep_memo_converted = []
        for item in rep_memo:
            s_t, a_t, r_t, s_t_plus_1 = item
            
            # 如果元素是 torch.Tensor，转换为 numpy 数组，再转换为列表
            s_t = s_t.numpy().tolist() if isinstance(s_t, torch.Tensor) else s_t
            a_t = a_t.numpy().tolist() if isinstance(a_t, torch.Tensor) else a_t
            r_t = r_t.numpy().tolist() if isinstance(r_t, torch.Tensor) else r_t
            s_t_plus_1 = s_t_plus_1.numpy().tolist() if isinstance(s_t_plus_1, torch.Tensor) else s_t_plus_1
            
            # 将转换后的元素添加到新列表中
            rep_memo_converted.append((s_t, a_t, r_t, s_t_plus_1))
        
        return rep_memo_converted

    # called after the warm-up period
    def build_initial_rep_memo_parameter_sharing(self):
        for wc in self.wc_list:
            self.rep_memo += wc.rep_memo.copy()
            # and clear workcenter's replay memory
            wc.replay_memory = []
            # 将 self.rep_memo 中的张量转换为列表
        rep_memo_converted = self.convert_rep_memo(self.rep_memo)

        # print('INITIALIZATION - replay_memory is:\n',len(self.rep_memo),\
        # tabulate(rep_memo_converted, headers = ['s_t','a_t','r_t','s_t+1']))
        print('input-size:', self.input_size, '\nreplay_memory_size:', len(self.rep_memo))
        print('---------------------------initialization accomplished-----------------------------')
        if self.log_info:
            table = tabulate(rep_memo_converted, headers=['s_t', 'a_t', 'r_t', 's_t+1'])
            # 记录日志
            logging.info('INITIALIZATION - replay_memory is:\n%s', table)
            logging.info('input-size: %s\nreplay_memory_size: %s', self.input_size, len(self.rep_memo))
            logging.info('---------------------------initialization accomplished-----------------------------')            

    def build_initial_rep_memo_independent(self):
        print('INITIALIZATION - replay_memory')
        for wc in self.wc_list:
            self.rep_memo[wc.wc_idx] += wc.rep_memo.copy()
            wc.replay_memory = []
            print(tabulate(self.rep_memo[wc.wc_idx], headers = ['s_t','a_t','s_t+1','r_t']))
            print('INITIALIZATION - size of replay memory:',len(self.rep_memo[wc.wc_idx]))
        print('---------------------------initialization accomplished-----------------------------')

    def update_rep_memo_parameter_sharing_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for wc in self.wc_list:
                self.rep_memo += wc.rep_memo.copy()
                # and clear workcenter's replay memory
                wc.rep_memo = []
            # and clear obsolete memory
            # print('not Truncate replay memory to size:',len(self.rep_memo), "time:", self.env.now)
            if len(self.rep_memo) > self.rep_memo_size:
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
                # print('Truncate replay memory to size:',len(self.rep_memo), "time:", self.env.now)
            yield self.env.timeout(self.routing_action_NN_training_interval*10)

    def update_rep_memo_independent_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for wc in self.wc_list:
                self.rep_memo[wc.wc_idx] += wc.rep_memo.copy()
                wc.rep_memo = []
            if len(self.rep_memo[wc.wc_idx]) > self.rep_memo_size:
                truncation = len(self.rep_memo[wc.wc_idx])-self.rep_memo_size
                self.rep_memo[wc.wc_idx] = self.rep_memo[wc.wc_idx][truncation:]
            yield self.env.timeout(self.routing_action_NN_training_interval*10)

    '''
    4. downwards are functions used in the training of DRL, including the dynamic training process
       dynamic training parameters update and the optimization funciton of ANN
       the class for builidng the ANN is at the bottom
    '''


    # print out the functions and classes used in the training
    def check_parameter(self):
        print('------------- Training Parameter Check -------------')
        print("Address seed:",self.address_seed)
        print('State Func.:',self.build_state.__name__)
        print('ANN:',self.routing_action_NN.__class__.__name__)
        print('------------- Training Parameter Check -------------')
        print('Discount rate:',self.discount_factor)
        print('Train feq: %s, Sync feq: %s'%(self.routing_action_NN_training_interval,self.routing_target_NN_sync_interval))
        print('Rep memo: %s, Minibatch: %s'%(self.rep_memo_size,self.minibatch_size))
        print('------------- Training Scenarios Check -------------')
        print("Configuration: {} work centers, {} machines".format(len(self.wc_list),len(self.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------')

    # train the action NN periodically
    def training_process_parameter_sharing(self):
        yield self.env.timeout(self.warm_up)
        #print(list(self.routing_action_NN.parameters()))
        for i in range(20):
            self.train()
        #print(list(self.routing_action_NN.parameters()))
        while self.env.now < self.span:
            self.train()
            yield self.env.timeout(self.routing_action_NN_training_interval)
        rep_memo_converted = self.convert_rep_memo(self.rep_memo)
        # print('Final replay_memory is:\n','size:',len(self.rep_memo),\
        # tabulate(rep_memo_converted, headers = ['s_t','a_t','r_t','s_t+1']))
        print('Final replay_memory is:\n','size:',len(self.rep_memo))
        # save the parameters of policy / action network after training
        # 确保文件夹存在
        os.makedirs(os.path.join(sys.path[0], 'routing_models'), exist_ok=True)
        torch.save(self.routing_action_NN.state_dict(), self.address_seed.format(sys.path[0]))
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))

    def training_process_independent(self):
        yield self.env.timeout(self.warm_up)
        for i in range(10):
            for wc in self.wc_list:
                self.train(wc.wc_idx)
        while self.env.now < self.span:
            for wc in self.wc_list:
                self.train(wc.wc_idx)
            yield self.env.timeout(self.routing_action_NN_training_interval)
        for wc in self.wc_list:
            print('FINAL - replay_memory of machine %s is:'%wc.wc_idx)
            print(tabulate(self.rep_memo[wc.wc_idx],headers = ['s_t','a_t','s_t+1','r_t']))
            address = self.address_seed.format(sys.path[0],str(wc.wc_idx))
            torch.save(self.routing_action_NN.network_dict[wc.wc_idx].state_dict(), address)
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))


    # synchronize the ANN and TNN, and some settings
    def update_training_setting_process(self):
        # one second after the initial training, so we can have a slightly better target network
        yield self.env.timeout(self.warm_up+1)
        while self.env.now < self.span:
            # synchronize the parameter of policy and target network
            self.routing_target_NN = copy.deepcopy(self.routing_action_NN)
            print('--------------------------------------------------------')
            print('the target network and epsilion are updated at time %s' % self.env.now)
            print('--------------------------------------------------------')
            if self.log_info:
                logging.info('--------------------------------------------------------')
                logging.info('the target network and epsilion are updated at time %s' % self.env.now)
                logging.info('--------------------------------------------------------')
            yield self.env.timeout(self.routing_target_NN_sync_interval)        # 每隔 250 同步一次

    # reduce the learning rate periodically
    def update_training_parameters_process(self):
        # one second after the initial training
        yield self.env.timeout(self.warm_up)
        reduction = (self.routing_action_NN.lr - self.routing_action_NN.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            # reduce the learning rate
            self.routing_action_NN.lr -= reduction
            self.epsilon -= 0.002
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.routing_action_NN.lr, self.env.now))
            print('--------------------------------------------------------')
            
            if self.log_info:
                logging.info('--------------------------------------------------------')
                logging.info('learning rate adjusted to {} at time {}'.format(self.routing_action_NN.lr, self.env.now))
                logging.info('--------------------------------------------------------')

    # reduce the learning rate periodically
    def update_learning_rate_process(self):
        # one second after the initial training
        yield self.env.timeout(self.warm_up)
        reduction = (self.lr - self.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            # reduce the learning rate
            self.lr -= reduction
            self.optimizer = optim.SGD(self.routing_action_NN.parameters(), lr = self.lr, momentum = 0.9)
            self.epsilon -= 0.01
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.lr, self.env.now))
            print('--------------------------------------------------------')


    # the function that draws minibatch and trains the action NN
    def train_DDQN(self):
        # print(".............TRAINING .............%s"%(self.env.now))
        if self.log_info:
            logging.info(".............TRAINING .............%s"%(self.env.now))
        """
        draw the random minibatch to train the network, randomly
        every element in the replay menory is a list [s_0, a_0, r_0, s_1]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        '''
        slice, and stack the 1D tensors to several 2D tensors (batches)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        # reshape so we can meet the requiremnt of input of instancenorm
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        sample_s1_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1,self.input_size)
        # reshape so we can use .gather() method
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.routing_action_NN.forward(sample_s0_batch)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)

        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)

        if self.log_info:
            logging.info('Q_0 is:\n%s'%Q_0)
            logging.info('a_0 is:\n%s'%sample_a0_batch)
            logging.info('current value is:\n%s'%current_value)

        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.routing_action_NN.forward(sample_s1_batch)
        Q_1_target = self.routing_target_NN.forward(sample_s1_batch)
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)

        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        # print('max value of Q_1_action is:\n', max_Q_1_action)

        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        # print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        # print('estimated value of next state is:\n', next_state_value)

        discounted_next_state_value = self.discount_factor * next_state_value
        # print('discounted next state value is:\n', discounted_next_state_value)

        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + discounted_next_state_value)
        #print('target value is:', target_value)
        '''
        the loss is difference between current state-action value and the target value
        '''
        # calculate the loss
        loss = self.routing_action_NN.loss_func(current_value, target_value)
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))

        # if self.log_info:
        #     logging.info('Q_1_action is:\n%s'%Q_1_action)
        #     logging.info('Q_1_target is:\n%s'%Q_1_target)
        #     logging.info('max value of Q_1_action is:\n%s'%max_Q_1_action)
        #     logging.info('max idx of Q_1_action is:\n%s'%max_Q_1_action_idx)
        #     logging.info('estimated value of next state is:\n%s'%next_state_value)
        #     logging.info('discounted next state value is:\n%s'%discounted_next_state_value)
        #     logging.info('loss is:\n%s'%loss)

        self.loss_record.append(float(loss))
        # clear the gradient
        self.optimizer.zero_grad()  # zero the gradient buffers
        # calculate gradient
        loss.backward(retain_graph=True)
        # set the optimizer
        # print('perform the optimization of parameters')
        # optimize the parameters
        self.optimizer.step()


    def loss_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5.5))
        # left half, showing the loss of training
        loss_record = fig.add_subplot(1,1,1)
        loss_record.set_xlabel('Iterations of training ('+r'$\times 10^3$'+')')
        loss_record.set_ylabel('Loss (error) of training')
        iterations = np.arange(len(self.loss_record))
        loss_record.scatter(iterations, self.loss_record,s=0.6,color='r', alpha=0.2)
        # moving average
        x = 50
        loss_record.plot(np.arange(x/2,len(self.loss_record)-x/2+1,1),np.convolve(self.loss_record, np.ones(x)/x, mode='valid'),color='k',label='moving average',zorder=3)
        # limits, grids,
        ylim_upper = 0.2
        ylim_lower = 0
        loss_record.set_xlim(0,len(self.loss_record))
        loss_record.set_ylim(ylim_lower,ylim_upper)
        xtick_interval = 1000
        loss_record.set_xticks(np.arange(0,len(self.loss_record)+1,xtick_interval))
        loss_record.set_xticklabels(np.arange(0,len(self.loss_record)/xtick_interval,1).astype(int),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        loss_record.set_yticks(np.arange(ylim_lower, ylim_upper+0.01, 0.01))
        loss_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        loss_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        loss_record.legend()
        # dual axis
        ax_time = loss_record.twiny()
        ax_time.set_xlabel('Time in simulation ('+r'$\times 10^3$'+', excluding warm up phase)')
        ax_time.set_xlim(self.warm_up,self.span)
        ax_time.set_xticks(np.arange(self.warm_up,self.span+1,xtick_interval*2))
        ax_time.set_xticklabels(np.arange(self.warm_up/xtick_interval,self.span/xtick_interval+1,2).astype(int),rotation=30, ha='left', rotation_mode="anchor", fontsize=8.5)
        loss_record.set_title("Routing Agent Training Loss / {}-machine per work centre test".format(int(len(self.m_list)/len(self.job_creator.wc_list))))
        fig.subplots_adjust(top=0.8, bottom=0.1, right=0.9)
        plt.show()
        # save the figure if required
        if 'save' in kwargs and kwargs['save']:
            # 获取目标目录路径
            save_dir = os.path.join(sys.path[0], "experiment_result")
            # 如果目录不存在，则创建目录
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            address = sys.path[0]+"//experiment_result//RA_loss_{}wc_{}m.png".format(len(self.job_creator.wc_list),len(self.m_list))
            fig.savefig(address, dpi=500, bbox_inches='tight')
            print('figure saved to'+address)
        return

    def reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5))
    # right half, showing the record of rewards
        reward_record = fig.add_subplot(1,1,1)
        reward_record.set_xlabel('Time')
        reward_record.set_ylabel('Reward')
        time = np.array(self.job_creator.rt_reward_record).transpose()[0]
        rewards = np.array(self.job_creator.rt_reward_record).transpose()[1]
        #print(time, rewards)
        reward_record.scatter(time, rewards, s=1,color='g', alpha=0.3, zorder=3)
        reward_record.set_xlim(0,self.span)
        reward_record.set_ylim(-1.1,1.1)
        xtick_interval = 2000
        reward_record.set_xticks(np.arange(0,self.span+1,xtick_interval))
        reward_record.set_xticklabels(np.arange(0,self.span+1,xtick_interval),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        reward_record.set_yticks(np.arange(-1, 1.1, 0.1))
        reward_record.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        reward_record.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        # moving average
        x = 50
        # print(len(time))
        reward_record.plot(time[int(x/2):len(time)-int(x/2)+1],np.convolve(rewards, np.ones(x)/x, mode='valid'),color='k',label="moving average")
        reward_record.legend()
        plt.show()
        # save the figure if required
        # fig.subplots_adjust(top=0.5, bottom=0.5, right=0.9)
        fig.subplots_adjust(right=0.9)

        if 'save' in kwargs and kwargs['save']:
            fig.savefig(sys.path[0]+"//experiment_result//RA_reward_{}wc_{}m.png".format(len(self.job_creator.wc_list),len(self.m_list)), dpi=500, bbox_inches='tight')
        return

'''
TEST network
'''
class build_network_TEST(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_TEST, self).__init__()
        # size of layers
        layer_1 = 64
        layer_2 = 64
        layer_3 = 48
        layer_4 = 32
        layer_5 = 16
        layer_6 = 16
        # FCNN
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, layer_6)
        self.fc7 = nn.Linear(layer_6, output_size)
        # activation functions
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        # Huber loss function
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        # print('original',x)
        x = self.instancenorm(x)
        # print('normalized',x)
        x = self.flatten(x)
        # print('flattened',x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.tanh(x)
        x = self.fc7(x)
        return x
    
class bulid_network_validated(nn.Module):
    def __init__(self, input_size, output_size):
        super(bulid_network_validated, self).__init__()
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.m_available_size = 9
        self.ttd_slack_size = 11
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_m_available = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(2),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_m_available, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        # print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_m_available = x[:,:, self.pt_size : self.m_available_size]
        x_ttd_slack = x[:,:, self.m_available_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_m_available = self.network[2](x_m_available)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_m_available, x_normed_ttd_slack, x_rest], dim=1)
        # print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
        #print('output',x)
        return x