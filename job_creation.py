import numpy as np
import random
import sys
sys.path
from tabulate import tabulate
import matplotlib.pyplot as plt
import logging
from tools import *
from collections import defaultdict
import pprint
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    filename='production.log',  # 将日志输出到文件
    filemode='w'  # 覆盖模式（'w'），每次运行清空文件；追加模式（'a'）则保留历史日志
)

'''mpc : 机器加工系数(Machine Processing Coefficient, MPC)表示为某道工序机器实际工作时间和机器期望加工时间的比值'''
class creation:
    def __init__ (self, env, span, machine_list, workcenter_list, pt_range, due_tightness, E_utliz, mpc_max, length_list,**kwargs):
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            print("Random seed of job creation is fixed, seed: {}".format(kwargs['seed']))
        self.log_info = True
        
        # environemnt and simulation span
        self.env = env
        self.span = span
        # all machines and workcenters
        self.m_list = machine_list
        self.wc_list = workcenter_list
        # the longest operaiton sequence passes all workcenters
        self.no_wcs = len(self.wc_list)
        self.no_machines = len(self.m_list)
        self.m_per_wc = int(self.no_machines / self.no_wcs)
        # the dictionary that records the details of operation and tardiness
        # operation record, path, wait time, decision points, slack change
        self.production_record = {}
        self.tardiness_record = {}
        # the reward record
        self.sqc_reward_record = []
        self.rt_reward_record = []
        # range of processing time
        self.pt_range = pt_range
        self.mpc_max = mpc_max
        # 使用 numpy 的 arange 生成候选数字列表 step = 0.05 从 1 到 mpc 
        # self.candidates = np.arange(1)
        self.candidates = np.arange(1, self.mpc_max, 0.05)
        # calulate the average processing time of a single operation
        self.avg_pt = np.average(self.pt_range) - 0.5
        # tightness factor of jobs
        self.tightness = due_tightness
        # expected utlization rate of machines
        self.E_utliz = E_utliz
        self.m_per_wc_ = length_list
        # generate a upscending seed for generating initial sequence, start from 0
        self.sequence_seed = np.arange(self.no_wcs)
        # set a variable to track the number of in-system number of jobs
        self.in_system_job_no = 0
        self.in_system_job_no_dict = {}
        self.index_jobs = 0
        # set lists to track the completion rate, realized and expected tardy jobs in system
        # 更新在agent_machine 的 update_global_info_progression
        self.comp_rate_list = [[] for m in self.m_list]
        self.comp_rate = 0
        self.realized_tard_list = [[] for m in self.m_list]
        self.realized_tard_rate = 0
        self.exp_tard_list = [[] for m in self.m_list]
        self.exp_tard_rate = 0
        # initialize the information associated with jobs that are being processed
        self.available_time_list = np.array([0 for m in self.m_list])
        self.release_time_list = np.array([self.avg_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.no_machines)
        self.next_wc_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.avg_pt for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.next_ttd_list = np.array([self.avg_pt*self.no_wcs for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        # and create an empty, initial array of sequence
        self.job_name_queue = []
        self.sequence_list = []

        # create the list to store the processing time mpc cmt, remaining processing time, due date of jobs
        # pt_list 是理想加工时间，mpc_list 是机器加工系数，cmt_list 是实际加工时间（包括了机器加工准备时间）
        self.pt_list = []
        self.mpc_list = []
        self.cmt_list = []
        self.remaining_pt_list = []
        self.remaining_mpc_list = []
        self.remaining_cmt_list = []

        self.create_time = []
        self.due_list = []
        # record the arrival and departure information
        self.arrival_dict = {}
        self.departure_dict = {}
        self.mean_dict = {}
        self.std_dict = {}
        self.expected_tardiness_dict = {}
        # decide the feature of new job arrivals
        # beta is the average time interval between job arrivals
        # let beta equals half of the average time of single operation
        # self.span：仿真的总时间。self.beta：作业到达的平均时间间隔。
        self.beta = self.avg_pt / (self.m_per_wc * self.E_utliz)
        # number of new jobs arrive within simulation
        self.total_no = np.round(self.span/self.beta).astype(int)
        # self.total_no = 100
        # print("beta: {}   total_no : {}".format(self.beta, self.total_no))
        # the interval between job arrivals by exponential distribution
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        # dynamically change the random seed to avoid extreme case
        if 'realistic_var' in kwargs and kwargs['realistic_var']:
            self.ptl_generation = self.ptl_generation_realistic
            self.realistic_var = kwargs['realistic_var']
        else:
            self.ptl_generation = self.ptl_generation_random
            # 如果不设计 随机种子 每次ptl都是一样的
        if 'random_seed' in kwargs and kwargs['random_seed']:
            interval = self.span/50
            print("set up random_seed = true {}".format(interval))
            self.env.process(self.dynamic_seed_change(interval))
        if 'hetero_len' in kwargs and kwargs['hetero_len']:
            pass
        if 'even' in kwargs and kwargs['even']:
            print("EVEN mode ON")
            #print(self.arrival_interval)
            self.arrival_interval = np.ones(self.arrival_interval.size)*self.arrival_interval.mean()
            #print(self.arrival_interval)
        self.initial_job_assignment()
        # start the new job arrival
        self.env.process(self.new_job_arrival())

    def initial_job_assignment(self):
        self.job_dict = generate_job_permutations(list(range(self.no_wcs)))
        self.job_store = [0 for _ in range(len(self.job_dict))]

        kind_job = random.randint(1, len(self.job_dict))
        
        # 随机选取 n 个不重复的键
        selected_keys = random.sample(list(self.job_dict.keys()), kind_job)
        # 为每个工件随机分配一个数量（1 到 self.no_wcs 之间）
        self.job_dict = generate_job_permutations(list(range(3)))
        quantities = [random.randint(1, self.no_wcs) for _ in range(len(selected_keys))]
        selected_keys = random.sample(list(self.job_dict.keys()), 5)
        quantities = [random.randint(3, 3) for _ in range(len(selected_keys))]
        # 构建选取的工件字典
        selected_jobs = {key: (self.job_dict[key], quantities[i]) for i, key in enumerate(selected_keys)}
        
        # 将 selected_jobs 转换为列表形式
        data = [
            {"job_name": int(key), "sqc": value[0], "quantity": value[1]}
            for key, value in selected_jobs.items()
        ]
        for item in data:
            self.job_store[item['job_name']] += item['quantity']
        # 按照 sqc 的第一个元素排序
        sorted_data = sorted(data, key=lambda x: x["sqc"][0])
        # 将 sorted_data 转换回字典形式
        sorted_jobs = {
            f"{item['job_name']}": (item["sqc"], item["quantity"])
            for item in sorted_data
        }
        
        if self.log_info:
            initial_info = [[job_name, perm, quantity] for job_name, (perm, quantity) in sorted_jobs.items()]
            # 使用 pprint.pformat 格式化字典
            formatted_dict = pprint.pformat(self.job_dict, indent=4)
            # 记录日志
            logging.info('job_dict Log:\n%s', formatted_dict)
            logging.info('job_store Log:\n%s', self.job_store)
            table = tabulate(initial_info, headers=['job_name.','sqc.','quantity'], tablefmt='pretty')
            logging.info('Initial Job Assignment:\n%s', table)
            logging.info('************************************')
        # 初始化工作中心字典
        wc_dict = defaultdict(list)

        # 将工件分配到对应的工作中心
        for item in sorted_data:
            wc_id = item["sqc"][0]  # 获取 sqc 的第一个元素作为工作中心 ID
            wc_dict[wc_id].append(item)
        for wc_idx, jobs in wc_dict.items():
            for job in jobs:
                sqc = np.array(job["sqc"])
                quantity = job["quantity"]
                job_name = job['job_name']

                # produce processing time of job, get corresponding remaining_pt_list
                ptl, mpc, cmt = self.ptl_generation()
                for _ in range(quantity):
                    self.sequence_list.append(sqc)
                    self.job_name_queue.append(job_name)
                    self.pt_list.append(ptl)
                    self.mpc_list.append(mpc)
                    self.cmt_list.append(cmt)
                    self.record_job_feature(self.index_jobs,ptl)
                    # reshape and rearrange the order of ptl to get remaining pt list
                    # remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[sqc]    #按照作业的加工顺序 sqc 重新排列加工时间。
                    # 将 split_by_slices 的返回值转换为 NumPy 数组
                    remaining_ptl = split_by_slices(ptl, self.m_per_wc_)[sqc]
                    self.remaining_pt_list.append(remaining_ptl)
                    remaining_mpc = split_by_slices(mpc, self.m_per_wc_)[sqc]
                    self.remaining_mpc_list.append(remaining_mpc)
                    remaining_cmt = split_by_slices(cmt, self.m_per_wc_)[sqc]
                    self.remaining_cmt_list.append(remaining_cmt)
                    # produce due date for job
                    avg_pt = ptl.mean()
                    # due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness))
                    # 更改数据 决定 due 的时间
                    due = np.round(avg_pt*self.no_wcs*np.random.uniform(1.2, self.tightness))
                    # record the creation time
                    self.create_time.append(0)
                    # add due date to due list, cannot specify axis
                    self.due_list.append(due)
                    # update the in-system-job number
                    self.record_job_arrival()
                    # operation record, path, wait time, decision points, slack change
                    self.production_record[self.index_jobs] = [[],[],[],{},[]]
                    
                    '''after creation of new job, add it to workcernter'''
                    # add job to system and create the data repository for job
                    self.wc_list[wc_idx].queue.append(self.index_jobs)
                    self.wc_list[wc_idx].job_name_queue.append(job_name)

                    # allocate the sequence of that job to corresponding workcenter's storage
                    # the added sequence is the one without first element, coz it's been dispatched
                    self.wc_list[wc_idx].sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
                    # allocate the processing time of that job to corresponding workcenter's storage
                    self.wc_list[wc_idx].pt_list.append(self.pt_list[self.index_jobs])
                    self.wc_list[wc_idx].mpc_list.append(self.mpc_list[self.index_jobs])
                    self.wc_list[wc_idx].cmt_list.append(self.cmt_list[self.index_jobs])
                    self.wc_list[wc_idx].remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
                    self.wc_list[wc_idx].remaining_mpc_list.append(self.remaining_mpc_list[self.index_jobs])
                    self.wc_list[wc_idx].remaining_cmt_list.append(self.remaining_cmt_list[self.index_jobs])

                    # allocate the due of that job to corresponding workcenter's storage
                    self.wc_list[wc_idx].due_list.append(self.due_list[self.index_jobs])

                    self.index_jobs += 1
            self.wc_list[wc_idx].routing_event.succeed()
        '''
        for wc_idx,wc in enumerate(self.wc_list): # for every work center
            np.random.shuffle(sqc_seed)
            sqc = np.concatenate([np.array([wc_idx]),sqc_seed[sqc_seed!=wc_idx]])
            random.shuffle(sqc)
            for m_idx,m in enumerate(wc.m_list): # for every machine inside this workcenter
                # allocate the job index to corrsponding workcenter's queue
                if flag:
                    flag = False
                    self.sequence_list.append(sqc)
                    # produce processing time of job, get corresponding remaining_pt_list
                    ptl = self.ptl_generation()
                    self.pt_list.append(ptl)
                    self.record_job_feature(self.index_jobs,ptl)
                    # reshape and rearrange the order of ptl to get remaining pt list
                    # remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[sqc]    #按照作业的加工顺序 sqc 重新排列加工时间。
                    # 将 split_by_slices 的返回值转换为 NumPy 数组
                    remaining_ptl = split_by_slices(ptl, self.m_per_wc_)[sqc]
                    print(sqc, remaining_ptl)
                    self.remaining_pt_list.append(remaining_ptl)
                    # produce due date for job
                    avg_pt = ptl.mean()
                    due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness))

                    # record the creation time
                    self.create_time.append(0)
                    # add due date to due list, cannot specify axis
                    self.due_list.append(due)
                    # update the in-system-job number
                    self.record_job_arrival()
                    # operation record, path, wait time, decision points, slack change
                    self.production_record[self.index_jobs] = [[],[],[],{},[]]
                    # print("**ARRIVAL: Job %s, time:%s, sqc:%s, pt:%s, due:%s"%(self.index_jobs, self.env.now, self.sequence_seed ,ptl, due))
                    
                    # after creation of new job, add it to workcernter
                    # add job to system and create the data repository for job
                    wc.queue.append(self.index_jobs)
                    # allocate the sequence of that job to corresponding workcenter's storage
                    # the added sequence is the one without first element, coz it's been dispatched
                    wc.sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
                    # allocate the processing time of that job to corresponding workcenter's storage
                    wc.pt_list.append(self.pt_list[self.index_jobs])
                    wc.remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
                    # allocate the due of that job to corresponding workcenter's storage
                    wc.due_list.append(self.due_list[self.index_jobs])

                    self.index_jobs += 1
            # after assigned the initial job to workcenter, activate its routing behavior
            if wc_idx == 0:
                wc.routing_event.succeed()'''

    def new_job_arrival(self):
        # main process
        while self.index_jobs < self.total_no:
            # draw the time interval betwen job arrivals from exponential distribution
            # The mean of an exp random variable X with rate parameter λ is given by:
            # 1/λ (which equals the term "beta" in np exp function)
            time_interval = self.arrival_interval[self.index_jobs]
            yield self.env.timeout(time_interval)

            quantity = random.randint(1, self.no_wcs)

            quantity = 1
            # 我们在这里假设 新添加的 工件 数量 只会是 1个
            selected_keys = random.sample(list(self.job_dict.keys()), 1)
            self.job_store[selected_keys[0]] += quantity
            job_name = selected_keys[0]

            self.sequence_seed = self.job_dict[selected_keys[0]]        
            # print(selected_keys, self.sequence_list)
            ptl, mpc, cmt = self.ptl_generation()
            # produce due date for job
            avg_pt = ptl.mean()
            due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness) + self.env.now)
            if self.log_info:
                logging.info("**ARRIVAL: %s Job : Job %s - Job %s, time:%s, sqc:%s, pt:%s, mpc:%s, cmt:%s, due:%s"% \
                             (quantity ,self.index_jobs,self.index_jobs + quantity - 1, self.env.now, self.sequence_seed ,ptl, mpc, cmt, due))
            for _ in range(quantity):
                # produce sequence of job, first shuffle the sequence seed
                np.random.shuffle(self.sequence_seed)
                self.sequence_list.append(np.copy(self.sequence_seed))
                self.job_name_queue.append(job_name)
                # produce processing time of job, get corresponding remaining_pt_list
                self.pt_list.append(ptl)
                self.mpc_list.append(mpc)
                self.cmt_list.append(cmt)
                self.record_job_feature(self.index_jobs,ptl)
                # reshape and rearrange the order of ptl to get remaining pt list  按照作业的加工顺序 sqc 重新排列加工时间
                remaining_ptl = split_by_slices(ptl, self.m_per_wc_)[self.sequence_seed]  # 将 split_by_slices 的返回值转换为 NumPy 数组
                self.remaining_pt_list.append(remaining_ptl)
                remaining_mpc = split_by_slices(mpc, self.m_per_wc_)[self.sequence_seed]   
                self.remaining_mpc_list.append(remaining_mpc)
                remaining_cmt = split_by_slices(cmt, self.m_per_wc_)[self.sequence_seed]    
                self.remaining_cmt_list.append(remaining_cmt)

                # record the creation time
                self.create_time.append(self.env.now)
                # add due date to due list, cannot specify axis
                self.due_list.append(due)

                '''after creation of new job, add it to workcernter'''
                # first workcenter of that job
                first_workcenter = self.sequence_seed[0]
                # add job to system and create the data repository for job
                self.record_job_arrival()
                # operation record, path, wait time, decision points, slack change, (最后一个工序实际完成时间， 实际得slack(大于0 表示延迟 小于0 表示在规定时间内完成))， job_name
                # 第6项在 machine 文件中的 record_slack_tardiness 最后一项保存了作业的实际完成时间
                self.production_record[self.index_jobs] = [[],[],[],{},[]]
                # add job and job name to workcenter
                self.wc_list[first_workcenter].queue.append(self.index_jobs)
                self.wc_list[first_workcenter].job_name_queue.append(job_name)
                # print('new arrival : ',job_name, self.index_jobs)
                # add sequence list to workcenter's storage
                self.wc_list[first_workcenter].sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
                # allocate the processing time of that job to corresponding workcenter's storage
                self.wc_list[first_workcenter].pt_list.append(self.pt_list[self.index_jobs])
                self.wc_list[first_workcenter].remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
                self.wc_list[first_workcenter].mpc_list.append(self.mpc_list[self.index_jobs])
                self.wc_list[first_workcenter].remaining_mpc_list.append(self.remaining_mpc_list[self.index_jobs])
                self.wc_list[first_workcenter].cmt_list.append(self.cmt_list[self.index_jobs])
                self.wc_list[first_workcenter].remaining_cmt_list.append(self.remaining_cmt_list[self.index_jobs])

                # allocate the due of that job to corresponding workcenter's storage
                self.wc_list[first_workcenter].due_list.append(self.due_list[self.index_jobs])
                # update index for next new job
                self.index_jobs += 1
                # and activate the dispatching of the work center
                try:
                    self.wc_list[first_workcenter].routing_event.succeed()
                except:
                    pass
                # print([m.remaining_pt_list for m in self.m_list])
            '''
            # produce sequence of job, first shuffle the sequence seed
            np.random.shuffle(self.sequence_seed)
            self.sequence_list.append(np.copy(self.sequence_seed))
            # produce processing time of job, get corresponding remaining_pt_list
            ptl = self.ptl_generation()
            self.pt_list.append(ptl)
            self.record_job_feature(self.index_jobs,ptl)
            # reshape and rearrange the order of ptl to get remaining pt list  按照作业的加工顺序 sqc 重新排列加工时间
            remaining_ptl = np.reshape(ptl,[self.no_wcs,self.m_per_wc])[self.sequence_seed]
            remaining_ptl = split_by_slices(ptl, self.m_per_wc_)[self.sequence_seed]  # 将 split_by_slices 的返回值转换为 NumPy 数组
            # print(self.remaining_pt_list, remaining_ptl)
            self.remaining_pt_list.append(remaining_ptl)
            # produce due date for job
            avg_pt = ptl.mean()
            due = np.round(avg_pt*self.no_wcs*np.random.uniform(1, self.tightness) + self.env.now)
            # record the creation time
            self.create_time.append(self.env.now)
            # add due date to due list, cannot specify axis
            self.due_list.append(due)
            logging.info("**ARRIVAL: Job %s, time:%s, sqc:%s, pt:%s, due:%s"%(self.index_jobs, self.env.now, self.sequence_seed ,ptl, due))
            # after creation of new job, add it to workcernter
            # first workcenter of that job
            first_workcenter = self.sequence_seed[0]
            # add job to system and create the data repository for job
            self.record_job_arrival()
            # operation record, path, wait time, decision points, slack change
            self.production_record[self.index_jobs] = [[],[],[],{},[]]
            # add job to workcenter
            self.wc_list[first_workcenter].queue.append(self.index_jobs)
            # add sequence list to workcenter's storage
            self.wc_list[first_workcenter].sequence_list.append(np.delete(self.sequence_list[self.index_jobs],0))
            # allocate the processing time of that job to corresponding workcenter's storage
            self.wc_list[first_workcenter].pt_list.append(self.pt_list[self.index_jobs])
            self.wc_list[first_workcenter].remaining_pt_list.append(self.remaining_pt_list[self.index_jobs])
            # allocate the due of that job to corresponding workcenter's storage
            self.wc_list[first_workcenter].due_list.append(self.due_list[self.index_jobs])
            # update index for next new job
            self.index_jobs += 1
            # and activate the dispatching of the work center
            try:
                self.wc_list[first_workcenter].routing_event.succeed()
            except:
                pass
            # print([m.remaining_pt_list for m in self.m_list])
            '''

    def ptl_generation_random(self):
        ptl = np.random.randint(self.pt_range[0], self.pt_range[1], size = [self.no_machines])
        mpc = np.random.choice(self.candidates, size = [self.no_machines])
        # 计算结果并保留两位小数
        cmt = np.around(ptl * mpc, 2)
        cmt = np.rint(ptl * mpc)
        mpc = np.around(mpc, 2)
        return ptl, mpc, cmt

    def ptl_generation_realistic(self):
        base = np.random.randint(self.pt_range[0], self.pt_range[1], [self.no_wcs,1]) * np.ones([self.no_wcs, self.m_per_wc])
        variation = np.random.randint(-self.realistic_var,self.realistic_var,[self.no_wcs, self.m_per_wc])
        #print(base,variation)
        ptl = (base + variation).clip(self.pt_range[0], self.pt_range[1])
        ptl = np.concatenate(ptl)
        return ptl

    def dynamic_seed_change(self, interval):
        while self.env.now < self.span:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
            # np.random.seed(10)
            # print('change random seed to {} at time {}'.format(seed,self.env.now))

    def change_setting(self,pt_range):
        print('Heterogenity changed at time',self.env.now)
        self.pt_range = pt_range
        self.avg_pt = np.average(self.pt_range)-0.5
        self.beta = self.avg_pt / (2*self.E_utliz)

    def get_global_exp_tard_rate(self):
        x = []
        for m in self.m_list:
            x = np.append(x, m.slack)
        rate = x[x<0].size / x.size
        return rate

    # this fucntion record the time and number of new job arrivals
    def record_job_arrival(self):
        self.in_system_job_no += 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.arrival_dict[self.env.now] += 1
        except:
            self.arrival_dict[self.env.now] = 1

    # this function is called upon the completion of a job, by machine agent  在agent_machine 中调用
    def record_job_departure(self):
        self.in_system_job_no -= 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.departure_dict[self.env.now] += 1
        except:
            self.departure_dict[self.env.now] = 1

    def record_job_feature(self,idx,ptl):
        self.mean_dict[idx] = (self.env.now, ptl.mean())
        self.std_dict[idx] = (self.env.now, ptl.std())

    # the sum of remaining processing time of all jobs in system
    # divided by the total number of machines on shop floor
    # is the estimation of waiting time for a new arrived job
    def get_expected_tardiness(self, ptl, due):
        sum_remaining_pt = sum([m.remaining_job_pt.sum() for m in self.m_list])
        expected_waiting_time = sum_remaining_pt / self.no_machines
        expected_processing_time = ptl.mean() * self.no_wcs
        expected_tardiness = expected_processing_time + expected_waiting_time + self.env.now - due
        #print('exp tard. of job {}: '.format(self.index_jobs),due, max(0, expected_tardiness))
        self.expected_tardiness_dict[self.index_jobs] = max(0, expected_tardiness)

    def build_sqc_experience_repository(self,m_list):
        # "grand" dictionary for replay memory
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        # for each machine to be controlled, build a sub-dictionary
        # because incomplete experience must be indexed by job's index
        for m in m_list:
            self.incomplete_rep_memo[m.m_idx] = {}
            self.rep_memo[m.m_idx] = []

    def output(self):
        print('job information are as follows:')
        job_info = [[i,self.sequence_list[i], self.pt_list[i], \
        self.create_time[i], self.due_list[i]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due']))
        print('--------------------------------------')
        return job_info

    def final_output(self):
        # information of job output time and realized tardiness
        output_info = []
        for item in self.production_record:
            output_info.append(self.production_record[item][5])
        job_info = [[i, self.job_name_queue[i], self.sequence_list[i], self.pt_list[i], self.mpc_list[i], self.cmt_list[i], self.create_time[i],\
        self.due_list[i], output_info[i][0], output_info[i][1]] for i in range(self.index_jobs)]
        # print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due','out','tard.']))
        self.generate_gannt_chart()
        if self.log_info:
            # 构建表格
            headers = ['idx.', 'name', 'sqc.', 'proc.t.','mpc','cmt.t.', 'in', 'due', 'out', 'tard.']
            table = tabulate(job_info, headers=headers, tablefmt='pretty')

            # 记录日志
            logging.info('Production Log:\n%s', table)
            logging.info('job_store Log:\n%s', self.job_store)
            logging.info('************************************')
        realized = np.array(output_info)[:,1].sum()
        exp_tard = sum(self.expected_tardiness_dict.values())

    def tardiness_output(self):
        # information of job output time and realized tardiness
        tard_info = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard_info.append(self.production_record[item][5])
        # now tard_info is an ndarray of objects, cannot be sliced. need covert to common np array
        # if it's a simple ndarray, can't sort by index
        dt = np.dtype([('output', float),('tardiness', float)])
        tard_info = np.array(tard_info, dtype = dt)
        tard_info = np.sort(tard_info, order = 'output')
        # now tard_info is an ndarray of objects, cannot be sliced, need covert to common np array
        tard_info = np.array(tard_info.tolist())
        tard_info = np.array(tard_info)
        output_time = tard_info[:,0]
        tard = np.absolute(tard_info[:,1])
        cumulative_tard = np.cumsum(tard)
        tard_max = np.max(tard)
        tard_mean = np.cumsum(tard) / np.arange(1,len(cumulative_tard)+1)
        tard_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return output_time, cumulative_tard, tard_mean, tard_max, tard_rate

    def record_printout(self):
        for i, item in enumerate(self.production_record):
            print(i ,self.production_record[item])

    def timing_output(self):
        return self.arrival_dict, self.departure_dict, self.in_system_job_no_dict

    def feature_output(self):
        return self.mean_dict, self.std_dict

    def all_tardiness(self):
        # information of job output time and realized tardiness
        tard = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard.append(self.production_record[item][5][1])
        #print(tard)
        tard = np.array(tard)
        mean_tardiness = tard.mean()
        #print(self.production_record)
        #print(tard)
        tardy_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return mean_tardiness, tardy_rate
    
    def generate_gannt_chart(self):
        # 获取作业数量
        extracted_data = self.production_record
        n_jobs = len(extracted_data)
        # 提取每个作业的数据
        data = {}
        for job_idx, job_data in extracted_data.items():
            operations = job_data[0]  # 操作时间点和加工时间
            machines = job_data[1]    # 机器索引
            job_name = job_data[6]  # 工件名字
            data[job_idx] = [operations, machines, job_name]
        # 使用 matplotlib 的 colormap 生成颜色列表
        colors = plt.cm.get_cmap('tab20').colors  # 获取 'tab20' 的所有颜色
        n_colors = len(colors)  # 颜色数量

        # 初始化绘图
        fig, ax = plt.subplots(figsize=(10, 6))

        # 遍历每个作业
        for job_idx, (operations, machines, job_name) in data.items():
            color = colors[job_idx % n_colors]  # 循环使用颜色
            for i, ((start_time, duration, cmt, flag), machine_idx) in enumerate(zip(operations, machines)):
                # 对时间值进行四舍五入，保留两位小数
                start_time_rounded = round(start_time, 2)
                duration_rounded = round(duration, 2)
                cmt_rounded = round(cmt, 2)

                if flag == 0:
                    # 正常绘制条形图
                    ax.barh(
                        machine_idx,  # 机器索引作为纵坐标
                        duration_rounded,  # 加工时间作为条形宽度
                        left=start_time_rounded,  # 操作开始时间作为条形起始位置
                        color=color,  # 作业颜色
                        edgecolor='black',  # 条形边框颜色
                        label=f'Job {job_name}_{job_idx}' if i == 0 else ""  # 仅第一次添加图例
                    )
                    # 在条形中间添加标签
                    label = (
                        f"Name {job_name}\n"  # 工件名字
                        f"Index {job_idx}\n"  # 工件序号
                        f"Op {i+1}\n"  # 工件序号和加工工序数
                        f"Start {start_time_rounded}\n"  # 开始时间
                        f"End {round(start_time_rounded + duration_rounded, 2)}\n"  # 结束时间
                        f"Dur {duration_rounded}"  # 加工时长
                    )
                    ax.text(
                        start_time_rounded + duration_rounded / 2,  # 标签的横坐标（条形中间）
                        machine_idx,  # 标签的纵坐标（机器索引）
                        label,  # 标签内容
                        ha='center',  # 水平居中
                        va='center',  # 垂直居中
                        fontsize=8,  # 字体大小
                        color='black'  # 字体颜色
                    )
                else:
                    # 将 cmt 拆分为 cmt - pt 和 pt
                    pt_rounded = duration_rounded  # 假设 duration 是 pt
                    cmt_minus_pt_rounded = round(cmt_rounded - pt_rounded, 2)  # cmt - pt 两位小数

                    # 绘制 cmt - pt 部分（红色边框）
                    ax.barh(
                        machine_idx,  # 机器索引作为纵坐标
                        cmt_minus_pt_rounded,  # cmt - pt 作为条形宽度
                        left=start_time_rounded,  # 操作开始时间作为条形起始位置
                        color='red',  # 作业颜色
                        edgecolor='black',  # 条形边框颜色
                    )
                    # # 在条形中间添加标签
                    # label = (
                    #     f"Name {job_name}\n"  # 工件名字
                    #     f"Index {job_idx}\n"  # 工件序号
                    #     f"Op {i+1}\n"  # 工件序号和加工工序数
                    #     f"Start {start_time_rounded}\n"  # 开始时间
                    #     f"End {round(start_time_rounded + cmt_minus_pt_rounded, 2)}\n"  # 结束时间
                    #     f"Dur {cmt_minus_pt_rounded}"  # 加工时长
                    # )
                    # ax.text(
                    #     start_time_rounded + cmt_minus_pt_rounded / 2,  # 标签的横坐标（条形中间）
                    #     machine_idx,  # 标签的纵坐标（机器索引）
                    #     label,  # 标签内容
                    #     ha='center',  # 水平居中
                    #     va='center',  # 垂直居中
                    #     fontsize=8,  # 字体大小
                    #     color='black'  # 字体颜色
                    # )
                    # 绘制 pt 部分
                    ax.barh(
                        machine_idx,  # 机器索引作为纵坐标
                        pt_rounded,  # pt 作为条形宽度
                        left=round(start_time_rounded + cmt_minus_pt_rounded, 2),  # 操作开始时间 + cmt - pt
                        color=color,  # 作业颜色
                        edgecolor='black',  # 条形边框颜色
                        label=f'Job {job_name}_{job_idx}' if i == 0 else ""  # 仅第一次添加图例
                    )
                    # 在条形中间添加标签
                    label = (
                        f"Name {job_name}\n"  # 工件名字
                        f"Index {job_idx}\n"  # 工件序号
                        f"Op {i+1}\n"  # 工件序号和加工工序数
                        f"Start {round(start_time_rounded + cmt_minus_pt_rounded, 2)}\n"  # 开始时间
                        f"End {round(start_time_rounded + cmt_rounded, 2)}\n"  # 结束时间
                        f"Dur {pt_rounded}"  # 加工时长
                    )
                    ax.text(
                        round(start_time_rounded + cmt_minus_pt_rounded + pt_rounded / 2, 2),  # 标签的横坐标（条形中间）
                        machine_idx,  # 标签的纵坐标（机器索引）
                        label,  # 标签内容
                        ha='center',  # 水平居中
                        va='center',  # 垂直居中
                        fontsize=8,  # 字体大小
                        color='black'  # 字体颜色
                    )

        # 获取所有 machines 的最大值
        max_machine = len(self.m_list)

        # 设置纵坐标标签（机器索引）
        ax.set_yticks(range(max_machine))
        ax.set_yticklabels([f'Machine {i}' for i in range(max_machine)])

        # 设置横坐标标签（时间）
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')

        # 添加图例
        ax.legend(loc='upper right')

        # 设置标题
        ax.set_title('Gantt Chart')

        # 显示网格
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # 显示图形
        plt.tight_layout()
        # 保存图片到本地文件
        plt.savefig('output.png')  # 保存为PNG格式，文件名是 output.png        
        plt.show()

