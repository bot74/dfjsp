import simpy
import sys
sys.path
import random
import numpy as np
import torch
from tabulate import tabulate
import sequencing
import routing
import logging

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='production.log',  # 将日志输出到文件
    filemode='w'  # 覆盖模式（'w'），每次运行清空文件；追加模式（'a'）则保留历史日志
)
# wc 的 self.m_list 是wc里面自己的数量
class workcenter:
    def __init__(self, env, index, m_list, *args, **kwargs):
        # initialize the environment of simulation
        self.env = env
        self.m_list = m_list
        self.m_no = len(self.m_list)
        self.m_idx_list = [m.m_idx for m in m_list]
        self.wc_idx = index
        # queue list stores the index of jobs and the name of jobs   在job_creation 中添加jobs
        self.queue = []
        self.job_name_queue = []
        # each workcenter will have an independent storage for each type of job information
        # initialize all job-related information storage as empty lists
        self.sequence_list = [] # sequence of all queuing jobs

        self.pt_list = [] # processing time
        self.mpc_list = []
        self.cmt_list = []
        self.remaining_pt_list = [] # remaining processing time
        self.remaining_mpc_list = []
        self.remaining_cmt_list = []

        self.due_list = [] # due for each job
        self.weight_list = [] # importance of each job
        self.routing_data = []
        # print out the information of initial jobs  log the information of wcs
        self.print_info = True
        self.log_info = True
        # initialize the routing(dispatching) event, for now it's untriggered
        # wait to be activated by job arrivals
        self.routing_event = self.env.event()
        # standby disctionary for storing incomplete experience for learning 存在于agent_machine.py中
        self.build_routing_experience = self.complete_experience_full
        self.incomplete_experience = {}
        # standby list for storing complete experience for learning
        # its content is taken by brain module periodically
        self.rep_memo = []
        # set the routing rule before start of simulation
        if 'rule' in kwargs:
            order = "self.job_routing = routing." + kwargs['rule']
            try:
                exec(order)
                print("workcenter {} uses {} routing rule".format(self.wc_idx, kwargs['rule']))
            except:
                print("Rule assigned to workcenter {} is invalid !".format(self.wc_idx))
                raise Exception
        else:
            # default routing rule is earliest available machine
            self.job_routing = routing.EA

    # this function should be called after __init__ to avoid deadlock
    # after the creation of all machine instances and initial job list for each machine
    # pass the list of machines to all machines so the shopfloor is established
    # the initial jobs are allocated through job_initialization module
    def initialization(self, job_creator):
        self.job_creator = job_creator
        self.dummy_pt = np.ones(self.m_no)*self.job_creator.avg_pt
        self.dummy_pt = self.job_creator.avg_pt
        # print("dummy_pt : ",self.dummy_pt)
        # initial information
        if self.print_info:
            print('work center {} contains machine {}'.format(self.wc_idx, self.m_idx_list))
            print('Initial %s jobs at workcenter %s are:'%(len(self.queue), self.wc_idx))
            job_info = [[self.job_name_queue[i],self.queue[i],self.sequence_list[i], self.pt_list[i], self.mpc_list[i], self.cmt_list[i], self.due_list[i]] for i in range(len(self.queue))]
            print(tabulate(job_info, headers=['job_name','idx.','sqc.','proc.t.','mpc','cmt.t.','due']))
            print('************************************')

        if self.log_info:
            job_info = [[self.job_name_queue[i], self.queue[i],self.sequence_list[i], self.pt_list[i], self.mpc_list[i], self.cmt_list[i],  self.due_list[i]] for i in range(len(self.queue))]
            logging.info('work center {} contains machine {}'.format(self.wc_idx, self.m_idx_list))
            logging.info('Initial %s jobs at workcenter %s are:' % (len(self.queue), self.wc_idx))
            headers = ['job_name', 'idx.','sqc.','proc.t.','mpc','cmt.t.','due']
            table = tabulate(job_info, headers=headers, tablefmt='pretty')

            # 记录日志
            logging.info('Production Log:\n%s', table)            
        '''
        When all initial data, control variables, jobs and machines are ready
        PROCESS the routing function
        '''
        # initialize the data (empty lists) of all constituent machines
        for m in self.m_list:
            m.state_update_all()
        """
        # # dispatch one initial job to all constituent machines
        # for i,m in enumerate(self.m_list):
        #     # get the first element of remaining pt list -> an array
        #     remaining_ptl = self.remaining_pt_list.pop(0)
        #     current_pt = remaining_ptl[0]
        #     # 估计的每一行取最大值，得到每个工作中心的最大加工时间。
        #     # 将这些最大值相加，得到作业在所有工作中心的最大加工时间总和
        #     estimated_slack_time = self.due_list[0] - self.env.now - np.sum(remaining_ptl.max(axis=1))
        #     remaining_ptl = np.delete(remaining_ptl, 0 ,axis=0) # remove the first element (row) of remaining_ptl, because it's now realized
        #     # add the information of job to machine agent's storage
        #     self.m_list[i].queue.append(self.queue.pop(0))
        #     self.m_list[i].sequence_list.append(self.sequence_list.pop(0))
        #     self.m_list[i].pt_list.append(self.pt_list.pop(0))
        #     # add slack, remaining.pt and due information to machines's storage
        #     self.m_list[i].slack_upon_arrival.append(estimated_slack_time)
        #     self.m_list[i].remaining_pt_list.append(remaining_ptl)
        #     self.m_list[i].due_list.append(self.due_list.pop(0))
        #     self.m_list[i].arrival_time_list.append(self.env.now)
        #     # after the job is routed to a machine, update the MACHINE's state
        #     self.m_list[i].state_update_after_job_arrival(15)
        # # after the initial assignment, process the real routing function
        # self.state_update_before_routing()"""
        # 对应下面的yield self.routing_event
        self.state_update_before_routing()
        self.env.process(self.routing())

    # The main process function that needs to be called in env.process()
    # the function run for one time when the new job / jobs join the queue
    def routing(self):
        while True:
            # program stucks here if event is not triggered
            # the event is triggerd by any job arrivals
            # 使用 yield self.routing_event 等待路由事件触发。当有新作业到达工作中心时，路由事件会被触发，继续执行后续代码。
            yield self.routing_event
            # could have more than 1 jobs join the queue at same time
            for j in range(len(self.queue)):

                # update current state  把 需要获得的信息都放到 self.state_update_before_routing() 中
                self.state_update_before_routing()
                # always dispatch the first remaining job
                # get the expected processing time of all remaining processes   从remaining_pt_list是已经排好序号的（按照加工顺序） 
                remaining_ptl = self.remaining_pt_list.pop(0)
                remaining_mpc = self.remaining_mpc_list.pop(0)
                remaining_cmt = self.remaining_cmt_list.pop(0)
                # get the list of processing time of that job on all constituent machines
                self.current_pt = remaining_ptl[0]   # 所以第一个就是当前工序两个机器的加工时间
                self.current_mpc = remaining_mpc[0]
                self.current_cmt = remaining_cmt[0]
                # calculate slack time based on expected waiting, processing time
                #estimated_slack_time = self.due_list[0] - self.env.now - self.average_waiting - np.sum(remaining_ptl.max(axis=1))
                # 对每个子数组求最大值 通过减去当前时间、最小等待时间和最大剩余处理时间，可以得到任务的剩余时间。  
                # 更新 添加了cmt 让slack time减去了剩下工序的最大时间
                self.estimated_slack_time = self.due_list[0] - self.env.now - self.least_waiting - np.sum(remaining_ptl.max(axis=1))
                self.estimated_slack_time = self.due_list[0] - self.env.now - self.least_waiting - np.sum(remaining_cmt.max(axis=1))
                # and remove pt of current operation on constituent machines 因为第一个已经分配给了一个特定的机器 所以删除
                self.remaining_ptl = np.delete(remaining_ptl, 0, axis=0) # remove the first element (row) of remaining_ptl, because it's dispatched to a specific machine 
                self.remaining_mpc = np.delete(remaining_mpc, 0, axis=0)
                self.remaining_cmt = np.delete(remaining_cmt, 0, axis=0)

                # select a machine
                # the returned value is machine's position in self.m_list

                # 对每个子数组求均值
                self.mean_values = [np.mean(arr) for arr in remaining_ptl]
                self.current_pt_mean = np.mean(self.current_pt)
                self.current_pt_std_dev = np.std(self.current_pt)
                
                if len(self.mean_values) > 0:
                    self.remaining_time_mean = np.mean(self.mean_values)
                    self.remaining_time_std_dev = np.std(self.mean_values)
                else:
                    self.remaining_time_mean = 0
                    self.remaining_time_std_dev = 0

                    # update the training data for routing
                self.routing_training_data = [
                    self.job_creator.in_system_job_no,  # Total number of system jobs
                    self.buffer_num,                    # Total number of buffers in work center
                    self.ur_mean,                       # Mean of machine utilization rates
                    self.ur_std_dev,                    # Standard deviation of machine utilization rates
                    self.current_pt_mean,               # Mean of current processing time of jobs
                    self.m_available_mean,              # Mean of machine available time
                    self.m_available_std_dev,            # Standard deviation of machine available time
                    self.m_cumulative_pt_mean,          # Mean of cumulative processing time of jobs
                    self.m_cumulative_pt_std_dev        # Standard deviation of cumulative processing time of jobs
                ]
                # logging.info(
                #     f"\nTotal number of system jobs: {self.job_creator.in_system_job_no}\n"
                #     f"Total number of buffers in work center: {self.buffer_num}\n"
                #     f"Mean of machine utilization rates: {self.ur_mean}\n"
                #     f"Standard deviation of machine utilization rates: {self.ur_std_dev}\n"
                #     f"Mean of current processing time of jobs: {self.current_pt_mean}\n"
                #     f"Mean of machine available time: {self.m_available_mean}\n"
                #     f"Standard deviation of machine available time: {self.m_available_std_dev}\n"
                #     f"Mean of cumulative processing time of jobs: {self.m_cumulative_pt_mean}\n"
                #     f"Standard deviation of cumulative processing time of jobs: {self.m_cumulative_pt_std_dev}"
                # )
                selected_machine_index = self.job_routing(self.queue[0], self.routing_data, self.current_pt, self.estimated_slack_time, self.wc_idx, np.sum(self.mean_values), len(self.remaining_ptl), self.routing_training_data)
                # print(self.queue,self.queue[0], self.routing_data, selected_machine_index, current_pt)
                # after assign this job to machine, the amount that machine's available time will increase by
                increased_available_time = self.current_pt[selected_machine_index]
                # # 考虑到机器的准备时间
                # if self.m_list[selected_machine_index].cmt_flag != self.job_name_queue[0]:
                #     increased_available_time += remaining_cmt[selected_machine_index]
                
                # print('ROUTING: Job %s to machine %s at time %s'%(self.queue[0], self.m_list[selected_machine_index].m_idx, self.env.now))
                if self.log_info:
                    job_info = [[self.queue, self.job_name_queue[0], self.queue[0], self.routing_data, self.machine_condition, [self.current_pt], [self.current_mpc], [self.current_cmt]]]
                    logging.info('ROUTING: Job_Name %s: Job %s to machine %s at time %s'%(self.job_name_queue[0] ,self.queue[0], self.m_list[selected_machine_index].m_idx, self.env.now))
                    headers = ['queue','job_name' ,'queue[0]','routing_data','machine_condition',"current_pt","current_mpc","current_cmt"]
                    table = tabulate(job_info, headers=headers, tablefmt='pretty')

                    # 记录日志
                    logging.info('ROUTING Log:\n%s', table)    
                
                # simply pop out the information of job, add to machine's storage
                self.m_list[selected_machine_index].queue.append(self.queue.pop(0))
                self.m_list[selected_machine_index].job_name_queue.append(self.job_name_queue.pop(0))
                self.m_list[selected_machine_index].sequence_list.append(self.sequence_list.pop(0))
                self.m_list[selected_machine_index].pt_list.append(self.pt_list.pop(0))
                self.m_list[selected_machine_index].mpc_list.append(self.mpc_list.pop(0))
                self.m_list[selected_machine_index].cmt_list.append(self.cmt_list.pop(0))

                # add slack, remaining.pt and due information to machines's storage
                self.m_list[selected_machine_index].slack_upon_arrival.append(self.estimated_slack_time)
                self.m_list[selected_machine_index].remaining_pt_list.append(self.remaining_ptl)
                self.m_list[selected_machine_index].remaining_mpc_list.append(self.remaining_mpc)
                self.m_list[selected_machine_index].remaining_cmt_list.append(self.remaining_cmt)

                self.m_list[selected_machine_index].due_list.append(self.due_list.pop(0))
                # 计算作业的实际到达时间  确保作业的实际到达时间不会早于机器的释放时间。
                self.m_list[selected_machine_index].arrival_time_list.append(max(self.env.now,self.m_list[selected_machine_index].release_time))
                #print('Jutified in time is:',self.m_list[selected_machine_index].release_time)
                # update the information of machine
                self.m_list[selected_machine_index].state_update_after_job_arrival(increased_available_time)
                try:
                    # relase from starvation
                    self.m_list[selected_machine_index].sufficient_stock.succeed()
                except:
                    # if not starving, just pass
                    pass
            # and delete this job from own queue
            # de-activate the routing(dispatching) event, wait to be activated again by new job arrivals
            # 这行代码的作用是停用当前的路由事件，并重新初始化一个路由事件。通过重新初始化路由事件，可以确保路由行为在下次有新作业到达时再次触发。
            self.routing_event = self.env.event()

    # this function accumulates data of consitituent machines
    def state_update_before_routing(self):
        # update condition of machines
        self.machine_condition = np.array([machine.sufficient_stock.triggered*1 for machine in self.m_list])

        # routing data: cumulative pt on machine, available time and que size  针对多个机器的
        self.routing_data = [machine.routing_data_generation(machine.sufficient_stock.triggered*1) for machine in self.m_list]

        # get the average waiting time to calculate the slack time  等于将要变成available的时间
        self.least_waiting = np.min(self.routing_data, axis=0)[1]
        # print(self.routing_data, self.least_waiting)

        avg = np.average(np.array(self.routing_data).clip(0),axis=0)
        self.average_workcontent = avg[0]
        self.average_waiting = avg[1]
        # print(self.wc_idx,self.routing_data,self.average_waiting,avg)

        # update the training data for routing
        self.buffer_num = sum(sublist[2] for sublist in self.routing_data)
        self.m_ur_list = [sublist[4] for sublist in self.routing_data]
        self.ur_mean = np.mean(self.m_ur_list)
        self.ur_std_dev = np.std(self.m_ur_list)
        self.m_available_list = [sublist[1] for sublist in self.routing_data]
        self.m_available_mean = np.mean(self.m_available_list)
        self.m_available_std_dev = np.std(self.m_available_list)
        self.m_cumulative_pt_list = [sublist[0] for sublist in self.routing_data]
        self.m_cumulative_pt_mean = np.mean(self.m_cumulative_pt_list)
        self.m_cumulative_pt_std_dev = np.std(self.m_cumulative_pt_list)




    # the function of building incomplete experience is in brain module
    # this function is called only if the routing_learning_event of constituent machine is triggered
    # when this function is called upon the completion of an operation
    def complete_experience_full(self, job_idx, slack_change, critical_level_R):
        # first update current state, extract data from constituent machines
        self.state_update_before_routing()

        # update the training data for routing
        self.routing_training_data = [
            self.job_creator.in_system_job_no,  # Total number of system jobs
            self.buffer_num,                    # Total number of buffers in work center
            self.ur_mean,                       # Mean of machine utilization rates
            self.ur_std_dev,                    # Standard deviation of machine utilization rates
            self.dummy_pt,                      # Mean of current processing time of jobs
            self.m_available_mean,              # Mean of machine available time
            self.m_available_std_dev,            # Standard deviation of machine available time
            self.m_cumulative_pt_mean,          # Mean of cumulative processing time of jobs
            self.m_cumulative_pt_std_dev        # Standard deviation of cumulative processing time of jobs
        ]
        
        # get the state at the time of job output
        # s_t = self.build_state(self.routing_data, self.dummy_pt, 0, self.wc_idx)
        s_t = self.build_state(self.routing_training_data, self.dummy_pt, 0, self.wc_idx)

        # calculate reward of action
        # if slack time of job reduces, reward in [-1,0], if slack time increases, in [0,1]
        r_t = torch.tensor(np.clip(slack_change*critical_level_R/20, -1, 1),dtype=torch.float)
        self.job_creator.rt_reward_record.append([self.env.now, r_t])

        logging.info("Job %s: Reward: %s  completed ex full %s  time %s", job_idx, r_t, s_t, self.env.now)

        #r_t = torch.tensor(np.clip(slack_change/40, -1, 1),dtype=torch.float)
        #print('build complete memory',job_idx,[r_t, s_t],self.env.now)
        # append reward and new state to corresponding incomplete experience
        self.incomplete_experience[job_idx] += [r_t, s_t]
        # and pop out this complete experience, add it to complete replay memory
        self.rep_memo.append(self.incomplete_experience.pop(job_idx))

    def complete_experience_global_reward(self, job_idx, slack_change, critical_level_R):
        self.state_update_before_routing()
        s_t = self.build_state(self.routing_training_data, self.dummy_pt, 0, self.wc_idx)
        self.incomplete_experience[job_idx] += [s_t]
