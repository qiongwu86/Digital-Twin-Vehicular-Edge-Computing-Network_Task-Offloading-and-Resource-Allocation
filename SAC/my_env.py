from __future__ import division
import numpy as np
import time
import random
import math
import collections

np.random.seed(123)

class Communication_Env:
    def __init__(self):
        self.h_bs = 10
        self.cofe = 0.2
        self.path_loss_indicator = 2
        self.Decorrelation_distance = 50
        self.BS_position = [0, 0, 10]
    
    def small_scale_path_loss(self, position, h_path):
        d1 = abs(position[0] - self.BS_position[0])
        d2 = abs(position[1] - self.BS_position[1])
        d3 = abs(position[2] - self.BS_position[2])
        distance = math.hypot(d1, d2, d3)
        return self.cofe*h_path + 1/(np.linalg.norm([distance,self.h_bs])**self.path_loss_indicator)


    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        d3 = abs(position_A[2] - self.BS_position[2])
        distance = math.hypot(d1, d2, d3)
        return 128.1 + 37.6 * np.log10(distance)# + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)

    def channel_gain(self,small_scale, large_scale):
        G_channel_gain = (abs(small_scale) ** 2) * large_scale
        return G_channel_gain
    
    def SINR(self, g_channel_gain, power, noise):
        Y_sinr = (g_channel_gain * power) / (noise)
        return Y_sinr

class Vehicle:
    def __init__(self, lane, position, velocity, task_num):
        self.lane = lane
        self.position = position
        self.velocity = velocity
        self.task_num = task_num
        self.task = []
    
    def task_generate(self):
        task_size = np.random.randint(100, 150, size=(1, self.task_num)) #Byte
        return task_size
     
    def vehicle_property(self):
        for i in self.task_generate():
            self.task.append(i)
        return {'v_lane': self.lane, 'position': self.position, 'velocity': self.velocity, 
                'compute_task': self.task}

class Environment:
    def __init__(self, lane_num, n_veh, width, task_num):
        self.lane_num = lane_num
        self.n_Veh = n_veh
        self.width = width
        self.channel = Communication_Env()

        self.pos_list = []
        self.trans_power = 200 #mW
        self.lane_width = 3.75
        self.L0 = 8
        self.noise = -110 #dBm
        self.noise1 = 10**(self.noise/10)
        self.t_slot = 0.1
        self.band_width = 50 #MHZ
        self.bs_max_fre = 100 #GHZ
        self.ve_max_fre = 5 #GHZ
        self.task_num = task_num
        self.t_limit = 0.1
        self.unit_cycle = 0.25
        self.delta_f = 0.2#np.round(np.random.rand() - 0.5, 2)

        self.vehicles = []
        #self.task_item = collections.OrderedDict()
    
    def add_new_vehicles(self, lane, position, velocity):
        self.vehicles.append(Vehicle(lane, position, velocity, self.task_num).vehicle_property())
    
    def add_new_vehicles_by_number(self):
        ini_pos = -6
        for i in range(self.n_Veh):
            self.pos_list.append(ini_pos)
            ini_pos += 4

        num = 0
        ind = 0
        while num < self.n_Veh:
            if ind < self.lane_num:
                lane_index = ind
                ind += 1
            else:
                ind = 0
                lane_index = ind
            y_pos = lane_index*self.lane_width + self.L0
            self.add_new_vehicles(lane_index, [self.pos_list[num], y_pos, 0], np.random.randint(10,15))
            num += 1

        self.task = []
        self.ve_v = np.zeros(len(self.vehicles))
        self.ve_l = np.zeros(len(self.vehicles))
        self.d_f = np.zeros(self.n_Veh)
        self.trans = np.zeros([self.n_Veh,self.task_num])
        self.exe_delay = np.zeros([self.n_Veh,self.task_num])
        self.tk_reshape = []
        self.tk_sac = []

        for i in range(len(self.vehicles)):
            self.task.append(self.vehicles[i]['compute_task'])
            self.ve_v[i] = self.vehicles[i]['velocity']
            self.ve_l[i] = self.vehicles[i]['position'][0]

        for sublist in self.task:
            for item in sublist: 
                self.tk_reshape.append(item)

        for sublist in self.tk_reshape:
            for item in sublist: 
                self.tk_sac.append(item)


        for i in range(self.n_Veh):
            self.d_f[i] = self.delta_f
        
        
        #for ve_id, ve in enumerate(self.vehicles):
        #    tk_size = ve['compute_task']
        #    for id, size in enumerate(tk_size):
        #        self.task_item.update({f'car_{ve_id}_{id}': [ve_id, id, size]})

        #initialized channel
        self.g_channel = np.zeros(len(self.vehicles))
        self.V2I_shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delt_distance = np.asarray([c['velocity']*self.t_slot for c in self.vehicles])
        real = np.random.normal(0, 1/2, len(self.vehicles))
        imag = np.random.normal(0, 1/2, len(self.vehicles))
        self.h_path = real + 1j*imag/math.sqrt(2)

    '车辆的位置更新'
    def vehicle_renew_position(self):
        i = 0
        while i < len(self.vehicles):
            v_position = self.vehicles[i]['position']
            velocity = self.ve_v[i]
            v_x = v_position[0] + velocity * self.t_slot
            #if v_position[0] < self.width/2 and v_x < self.width/2:
            self.vehicles[i]['position'][0] = v_x
            self.ve_l[i] = self.vehicles[i]['position'][0]
            #else:
            #    self.vehicles[i]['position'][0] = -self.width/2
            #    self.ve_l[i] = -self.width/2
            i += 1

    def renew_channel(self):
        self.V2I_path_loss = np.zeros(len(self.vehicles))
        self.large_scaled = np.zeros(len(self.vehicles))
        self.small_scaled = np.zeros(len(self.vehicles))
        self.V2I_shadowing = self.channel.get_shadowing(self.delt_distance, self.V2I_shadowing)

        for i in range(len(self.vehicles)):
            self.V2I_path_loss[i] = self.channel.get_path_loss(self.vehicles[i]['position'])
            self.small_scaled[i] = self.channel.small_scale_path_loss(self.vehicles[i]['position'], self.h_path[i])
            self.h_path[i] = self.small_scaled[i]
        
        self.large_scaled = self.V2I_path_loss + self.V2I_shadowing
    
    def R_V2I(self):
        self.V2I_rate = []
        for i in range(len(self.vehicles)):
            self.g_channel[i] = self.channel.channel_gain(self.small_scaled[i], self.large_scaled[i])
            self.g_channel[i] = 10**(self.g_channel[i]/10)
            y_sinr = self.channel.SINR(self.g_channel[i], self.trans_power/1000, self.noise1)
            v2i_rate = self.band_width*np.log2(1 + y_sinr)
            self.V2I_rate.append(v2i_rate)    # transmission rate from V to BS
    
    def delay(self, action):
        #exe_delay = np.array([])
        com_tk = np.reshape(self.task, [self.n_Veh,self.task_num])
        for i in range(len(self.vehicles)):
            rate_vb = self.V2I_rate[i]
            for j in range(self.task_num):
                task_size = com_tk[i][j]
                #uninstall to bs for processing
                T_tr = (action[i][j]*task_size)/rate_vb
                self.trans[i][j] = T_tr

                T_esm = (action[i][j]*task_size*self.unit_cycle)/(action[i][j+self.task_num]*1000)
                fenmu = (action[i][j+self.task_num]*1000)*((action[i][j+self.task_num]+self.delta_f)*1000)
                delta_T = -(action[i][j]*task_size*self.unit_cycle*self.delta_f*1000)/fenmu
                T_cmp = T_esm + delta_T
                T_dl = T_tr + T_cmp

                #local process
                T_lc = ((1-action[i][j])*task_size*self.unit_cycle)/(self.ve_max_fre*1000)
                
                T_total = max(T_dl, T_lc)
                self.exe_delay[i][j] = T_total
        #self.com_delay = exe_delay

        return  self.exe_delay
    
    def get_state(self):
        tk_reshape = np.reshape(self.task, [self.n_Veh, self.task_num])
        v_reshape = np.reshape(self.ve_v, [self.n_Veh, 1])
        l_reshape = np.reshape(self.ve_l, [self.n_Veh, 1])
        g_reshape = np.reshape(self.g_channel, [self.n_Veh, 1])
        delta_f_reshape = np.reshape(np.array([self.delta_f for x in range(self.n_Veh)]), [self.n_Veh, 1])
        cur_state = np.concatenate((tk_reshape, l_reshape, g_reshape),axis=1)
        return cur_state
    
    def act_for_training(self, actions):
        action_temp = actions.copy()
        reward = self.reward(action_temp)
        self.vehicle_renew_position()
        self.renew_channel()
        self.R_V2I()
        s_next = self.get_state()
        return s_next, reward
    
    def act_for_testing(self, actions):
        action_temp = actions.copy()
        
        mean_reward = np.mean(self.reward(action_temp))
        self.vehicle_renew_position()
        self.renew_channel()
        self.R_V2I()
        s_next = self.get_state()
        return s_next, mean_reward
    
    def reward(self, action):
        reward = np.array([])
        cmp_res = np.zeros([self.n_Veh,self.task_num])
        T_exe = self.delay(action)
        delay_limit = np.array([self.t_limit for x in range(self.task_num)])

        for i in range(self.n_Veh):
            for j in range(self.task_num):
                cmp_res[i][j] = action[i][j+self.task_num]
        
        used_fre = np.sum(cmp_res)+self.n_Veh*self.task_num*self.delta_f
        alpha = 0.5

        #if used_fre >= 0:
        for i in range(self.n_Veh):
            v_reward =10*np.mean(T_exe[i] - self.t_limit) - alpha*(np.sum(cmp_res[i]+self.delta_f)/self.n_Veh - self.bs_max_fre/self.n_Veh)              
            reward = np.append(reward, v_reward)
        return reward
    
    def make_new_game(self):
        self.vehicles = []
        self.add_new_vehicles_by_number()
        self.renew_channel()
        self.R_V2I()
        s = self.get_state()
        return s