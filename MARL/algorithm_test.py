import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(__file__))

from marl_train import Runner, Agent
from setting import arg
from my_env import Environment
from maddpg import MADDPG


def maddpg_test():
        agents = []
        for i in range(args.n_agents):
            Agent.policy = MADDPG(args, i).load_model()
            agent = Agent(i, args)
            agents.append(agent)
        returns = []
        t_tk = []
        alloc_tk = []
        for episode in range(args.test_episodes):
            # reset the environment
            s = env.make_new_game()
            rewards = 0
            time_tk = 0
            allocation_tk = 0
            for time_step in range(args.test_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for j in range(args.n_agents):
                    for i in range(args.task_num):    
                        actions[j][i+args.task_num] = actions[j][i+args.task_num]/3*100
                s_next, r = env.act_for_testing(actions)
                rewards += r
                time_tk += np.mean(env.exe_delay)
                rec = 0
                for i in range(env.n_Veh):
                    for j in range(env.task_num):
                        rec += actions[i][j+args.task_num]
                allocation_tk = rec
                s = s_next

            t_tk.append(time_tk/args.test_episode_len)
            alloc_tk.append(allocation_tk)
            rewards = rewards/args.test_episode_len
            returns.append(rewards)
        #print(actions)
        #plt.plot(range(len(returns)),returns)
        #plt.show()
        print('\nReturns is', sum(returns) / args.test_episodes)
        print('\nTime consumption for computing tasks:', sum(t_tk)/args.test_episodes)
        print('\nResources for computation tasks:', sum(alloc_tk)/args.test_episodes)
        #return sum(returns) / args.evaluate_episodes

if __name__ == '__main__':
    # get the params
    args = arg()
    env = Environment(args.lane_num, args.n_agents, args.width, args.task_num)
    maddpg_test()