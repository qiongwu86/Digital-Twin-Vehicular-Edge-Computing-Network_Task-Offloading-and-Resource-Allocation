from tqdm import tqdm
from maddpg import MADDPG
from Replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(0.01, 1.0, self.args.n_output).astype(np.float16)
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_net(inputs).squeeze(0)
            u = pi.cpu().numpy()
            # print('{} : {}'.format(self.name, pi))
            noise = noise_rate * self.args.action_bound * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, 0.01, 1.0)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name +'/' + 'marl_n_%d_-0.2'%self.args.n_agents
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.make_new_game()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            #for i in range(self.args.n_agents, self.args.n_players):
            #    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            for j in range(self.args.n_agents):
                for i in range(self.args.task_num):    
                    actions[j][i+self.args.task_num] = actions[j][i+self.args.task_num]/3*100
            
            s_next, r = self.env.act_for_training(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episodes')#+ str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                plt.close('all')
            self.noise = max(0.05, self.noise - 0.0000005)
            #K = int(np.floor(time_step / self.episode_limit))
            #self.epsilon = ((1-(1e-4))**K)*self.epsilon
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            np.save(self.save_path + '/returns.pkl', returns)   
        plt.figure()
        plt.plot(range(len(returns)), returns)
        plt.xlabel('episodes')
        plt.ylabel('average returns')
        plt.show()

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.make_new_game()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                #self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                #for i in range(self.args.n_agents, self.args.n_players):
                #    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                for j in range(self.args.n_agents):
                    for i in range(self.args.task_num):    
                        actions[j][i+self.args.task_num] = actions[j][i+self.args.task_num]/3*100
                s_next, r = self.env.act_for_testing(actions)
                rewards += r
                s = s_next
            rewards = rewards/self.args.evaluate_episode_len
            returns.append(rewards)
        print('\nReturns is', sum(returns) / self.args.evaluate_episodes)#rewards)
        return sum(returns) / self.args.evaluate_episodes