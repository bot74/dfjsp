import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate

import agent_machine
import agent_workcenter
import job_creation
import logging
import random

from tools import generate_length_list, get_group_index
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='production.log',  # 将日志输出到文件
    filemode='w'  # 覆盖模式（'w'），每次运行清空文件；追加模式（'a'）则保留历史日志
)

"""
THIS IS THE MODULE FOR TRAINING
"""

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, m_per_wc, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        # print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc[i])]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc[i]
        # print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
        [5,25], 2, 0.9, 1.2, m_per_wc, random_seed = True)
        # self.job_creator.output()

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = get_group_index(m_per_wc, i)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5: set up the brains for workcenters'''

        '''STEP 6: run the simulaiton'''
        env.run()
        self.job_creator.final_output()
        self.job_creator.record_printout()

# create the environment instance for simulation
env = simpy.Environment()
# create the shop floor instance
span = 100
m = 6
wc = random.randint(int(m/3),int(m/2))  # wc 的数量范围是 [2, 3]
wc = 3

length_list = generate_length_list(m, wc)

length_list = [2, 2, 2]
print(length_list)
spf = shopfloor(env, span, m, wc, length_list)

