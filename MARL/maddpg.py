import torch
import os
from ac_network import actor_network, crtic_network

class MADDPG:
    def __init__(self, arg, agent_id):
        self.arg = arg
        self.agent_id = agent_id
        self.train_step = 0

        self.actor_net = actor_network(arg)
        self.critic_net = crtic_network(arg)

        self.actor_target_net = actor_network(arg)
        self.critic_target_net = crtic_network(arg)

        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr= arg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr= arg.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.arg.save_dir):
            os.mkdir(self.arg.save_dir)
        # path to save the model
        self.model_path = self.arg.save_dir + '/' + self.arg.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'marl_n_%d_-0.2'%self.arg.n_agents
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型(在训练时注释)
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_net.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_net.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
        #    print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
        #                                                                  self.model_path + '/actor_params.pkl'))
        #    print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
        #                                                                   self.model_path + '/critic_params.pkl'))

    
    def soft_update_target_network(self):
        
        for target_param, param in zip(self.actor_target_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_((1-self.arg.tau)*target_param.data + self.arg.tau*param.data)
        
        for critic_target_param, critic_param in zip(self.critic_target_net.parameters(), self.critic_net.parameters()):
            critic_param.data.copy_((1-self.arg.tau)*critic_target_param.data + self.arg.tau*critic_param.data)
    
    def train(self,transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.arg.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.arg.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_net(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_net(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_net(o_next, u_next).detach()

            target_q = (r.unsqueeze(1) + self.arg.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_net(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_net(o[self.agent_id])
        actor_loss = - self.critic_net(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.arg.save_rate == 0:
            self.save_model()
        self.train_step += 1

    def save_model(self):
        #num = str(train_step // self.arg.save_rate)
        model_path = os.path.join(self.arg.save_dir, self.arg.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'marl_n_%d_-0.2'%self.arg.n_agents)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d'%self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_net.state_dict(), model_path + '/actor_params.pkl')
        torch.save(self.critic_net.state_dict(),  model_path + '/critic_params.pkl')
    
    def load_model(self):
        #if os.path.exists(self.model_path + '/actor_params.pkl'):
        self.actor_net.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
        self.critic_net.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
#        print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
#                                                                          self.model_path + '/actor_params.pkl'))
#        print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
#                                                                           self.model_path + '/critic_params.pkl'))
        self.actor_net.eval()
        self.critic_net.eval()