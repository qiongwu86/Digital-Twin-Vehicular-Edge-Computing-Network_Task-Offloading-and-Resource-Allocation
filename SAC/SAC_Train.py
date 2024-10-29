import numpy as np
import os
import scipy.io
import my_env
from RL_SAC import SAC_Trainer
from  RL_SAC import ReplayBuffer
import matplotlib.pyplot as plt

########################SETTING######################
lane_num = 3
n_veh = 5
width = 120
task_num = 3

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'model/sac_model/n_%d'%n_veh
model_path = label + '/agent'
data_path = 'data/sac_data/n_%d'%n_veh

env = my_env.Environment(lane_num, n_veh, width, task_num)
env.make_new_game()

n_episode = 200
n_step_per_episode = 100
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

#####################################################

def get_State(env, ind_episode=1., epsi=0.02):
    D_tk = env.tk_sac
    delta_f = env.d_f
    #vehicle_v = env.ve_v
    vehicle_l = env.ve_l
    vehicle_G = env.g_channel
    return np.concatenate((D_tk, delta_f , vehicle_l, vehicle_G, [ind_episode, epsi]))
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
## Initializations ##
# ------- characteristics related to the network -------- #
batch_size = 64
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(get_State(env=env))
n_output = task_num*2
action_range = 1.0
# --------------------------------------------------------------
#agent = SAC_Trainer(alpha, beta, n_input, tau, gamma, 12 ,memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, 2, 'OU')
replay_buffer_size = 2e5#1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
agent = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
frame_idx = 0
explore_steps = 0 # for random action sampling in the beginning of training
## Let's go
if IS_TRAIN:
    done = 0
    record_reward_average = []

    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final
        record_reward = np.zeros([n_step_per_episode], dtype=np.float16)

        #env.make_new_game()
        if i_episode % 20 == 0:
            env.vehicle_renew_position() # update vehicle position
            env.renew_channel()
            env.R_V2I()


        state_old_all = []
        #for i in range(n_veh):
        state = get_State(env, i_episode/(n_episode-1), epsi)
        state_old_all.append(state)

        average_reward = 0
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, task_num*2], dtype=np.float16)  # sub, power
            # receive observation
            if frame_idx > explore_steps:
                action = agent.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            else:
                action = agent.policy_net.sample_action()
            action = np.clip(action, 0.01, 1)
            action_all.append(action)
            # All the agents take actions simultaneously, obtain reward, and update the environment
            for i in range(n_veh):
                for j in range(task_num):
                    action_all_training[i, j] = action[j]
                    action_all_training[i, j+task_num] = ((action[j+task_num])/3 ) * 100  # chosen RB
            action_channel = action_all_training.copy()
            _, train_reward = env.act_for_training(action_channel)

            record_reward[i_step] = np.mean(train_reward)
            # get new state
            #for i in range(n_veh):
            state_new = get_State(env, i_episode / (n_episode - 1), epsi)
            state_new_all.append((state_new))


            # taking the agents actions, states and reward
            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                           train_reward, np.asarray(state_new_all).flatten(), done)

            # agents take random samples and learn
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    _ = agent.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*n_output)

            # old observation = new_observation
            state_old_all = state_new_all
            frame_idx += 1
        average_reward = np.mean(record_reward)
        record_reward_average.append(average_reward)
        print('step:', i_episode, 'reward', average_reward)

        if (i_episode+1) % 100 == 0 and i_episode != 0:
            agent.save_model(model_path)

    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    np.save(label + '/sac_returns.pkl', record_reward_average)
    plt.figure(2)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


    print('Training Done. Saving models...')
