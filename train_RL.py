import os

from torch import argmax, from_numpy
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import cv2
from random import random
import datetime
import time
from sys import platform

from cnn.basic_agent import BasicAgent

from constants import actions
import output

if platform == "win32":
    # Remove drone-v0 from registry
    import drone_remove
    drone_remove
    
from gym_drone.envs.drone_env import DroneEnv


def init_environment(map_file='map.csv', stations_file='bs.csv'):
    env = DroneEnv()

    # Get relevance map
    rel_map = np.genfromtxt(map_file, delimiter=';')
    #rel_map[0, 0] = 0
    #np.savetxt(map_file, rel_map, delimiter=';', fmt='%.1f')
    env.get_map(rel_map)

    # Get base stations
    base_stations = np.genfromtxt(stations_file, delimiter=';', dtype='int')
    base_stations = np.zeros_like(env.relevance_map)
    mid_x, mid_y = int(base_stations.shape[0]/2), int(base_stations.shape[1]/2)
    base_stations[mid_x, mid_y] = 100
    env.get_bases(base_stations)
    return env


def train_RL(episodes, iterations, replace_iterations, env, action_epsilon, 
             epsilon_decrease, batch_size, repmem_limit, path):
    tic_tic_tic = time.perf_counter()

    log_dir = 'runs'
    exp_f = os.path.join(log_dir, "exp{0}".format(get_exp_number(log_dir)))
    os.makedirs(exp_f, exist_ok=True)
    
    writer = SummaryWriter(log_dir=exp_f)
    #    Initialization
    agent = BasicAgent(actions, log_step=100, sw=writer)
    
    # agent.model = load_model("drone_model_2.pth")
    replay_memory = []
    #
    iter_counts = 0
    df = pd.DataFrame(columns=['Episode', 'Episode duration', 
                               'Number of steps', 'Total reward'])

    for i in range(episodes):
        # env.render(show=True)
        # the current state is the initial state
        state_matrix, cameraspot = env.reset()
        cs = get_current_state(env,state_matrix, cameraspot)
        done = False
        cnt = 0     # number of moves in an episode
        total_reward = 0
        tic_tic = time.perf_counter()
        total_a_dur = 0
        while not done:
            tic = time.perf_counter()
            
            env.render(show=True )
                       #i > 990)
            cnt += 1
            iter_counts += 1
            # select random action with eps probability or select action from model
            a, a_type = select_action(agent.model, cs, action_epsilon, agent.device)
            # update epsilon value taking into account the number of iterations
#            action_epsilon = update_epsilon(action_epsilon, epsilon_decrease, iter_counts)
            observation, reward, done, _ = env.step(a)
            if cnt>iterations:
                done = True
            total_reward += reward
            # if done and cnt < 200:
            #     reward = -1000
            
            # Getting new state
            state_matrix, _, cameraspot = observation
            new_state = get_current_state(env, state_matrix, cameraspot)
            replay_memory.append((cs, a, new_state, reward, done))
               
            # training the model after batch_size iterations
            if iter_counts % batch_size == 0:
                replay_memory=replay_memory[-repmem_limit:]
                data = np.random.permutation(np.array(replay_memory, dtype=object))[:batch_size]
                # train_qnet(model, data)
                agent.train(data)
                if agent.train_iterations % replace_iterations == 0:
                    agent.replace_target_network()
                
            cs = new_state
            toc = time.perf_counter()
            a_dur = toc - tic
            total_a_dur = total_a_dur + a_dur

        toc_toc = time.perf_counter()    
        ep_dur = toc_toc - tic_tic 
        total_dur = toc_toc - tic_tic_tic
        
        df.loc[i] = {'Episode': i, 'Episode duration': ep_dur,
                     'Number of steps': cnt, 'Total reward': total_reward}
        writer.add_scalar('episode duration', ep_dur, i)
        writer.add_scalar('number of steps', cnt, i)
        writer.add_scalar('total reward', total_reward, i)
        
        if (i+1) % 10 == 0:
            output.save_results(i, path, agent.model, df)
            output.save_info_file(path[1], episodes, repmem_limit, i+1, total_dur)
        output.print_episode_info(i, total_reward, cnt, ep_dur, total_a_dur)

    writer.close()
    return df


def select_action(model, cs, action_epsilon, device):
    if random() > action_epsilon:
        x = from_numpy(np.stack(cs)).unsqueeze(dim=0)
        pred = model(x.to(device))
        act_type = 'Model'
        position = argmax(pred, dim=1)
        return position.item(), act_type
    act = list(actions.keys())
    act_type = 'Random'
    return np.random.choice(act), act_type


def get_current_state(env, state_matrix, camera):
    full_map = env.relevance_map
    state_matrix = cv2.resize(state_matrix, full_map.shape) / env.max_battery
    return np.stack((state_matrix, full_map))


def get_exp_number(folder='runs'):
    ns = [int(f[3:]) for f in os.listdir(folder) if f.startswith('exp') and str.isdigit(f[3:])]
    return 1 + (max(ns) if len(ns) else 0)


if __name__ == '__main__':
    
    # Create a folder to save results
    today = datetime.datetime.today().strftime("%Y-%m-%d-%H.%M")
    path = output.create_dir(today)
    
    # PARAMS
    # episodes, iterations, env, action_epsilon, epsilon_decrease, batch_size
    m_file = "ones.csv"
    #m_file = "map_old.csv"
    env = init_environment(map_file=m_file)
    action_eps = 0.6

    batch_size = 16
    replace_iter = 32
    replay_memory_size = 1000
    iterations = 400
    epsilon_decrease = 0
    # How many episodes run: 
    episodes = 10
    
    table = train_RL(episodes, iterations, replace_iter, env,
                     action_eps, epsilon_decrease, batch_size,
                     replay_memory_size, path)    
    
    env.close() 
