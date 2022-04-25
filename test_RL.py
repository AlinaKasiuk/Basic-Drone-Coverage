from torch import argmax, from_numpy, load as torch_load
import numpy as np
import cv2
from random import random
from sys import platform

from cnn.structure import DroneQNet

from constants import IMG_H, IMG_W, actions

if platform == "win32":
    # Remove drone-v0 from registry
    import drone_remove
    drone_remove
    
from gym_drone.envs.drone_env import DroneEnv
from cnn.basic_agent import BasicAgent



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


def load_model(path):
    model = DroneQNet(2, IMG_W, IMG_H, len(actions))
    model.load_state_dict(torch_load(path))
    model.eval()
    model.double()
    return model
    

def get_current_state(state_matrix, camera):
    state_matrix = cv2.resize(state_matrix, (32, 32)) / 100
    resize_camera = cv2.resize(env.get_part_relmap_by_camera(camera), state_matrix.shape)
    return np.stack((state_matrix, resize_camera))


def test_trained_net(env, iterate=50, model_path="drone_model.pth"):
    agent = BasicAgent(actions)
    model = load_model(model_path)

    state_matrix, cameraspot = env.reset()
    cs = get_current_state(state_matrix, cameraspot)

    for i in range(iterate):
        env.render(show=True)
               #    i > 190)
        a, a_type = select_action(model, cs, 0, agent.device)
        observation, reward, done, _ = env.step(a)

        state_matrix, _, cameraspot = observation
        cs = get_current_state(state_matrix, cameraspot)


if __name__ == '__main__':
    
    m_file = "ones.csv"
    env = init_environment(map_file=m_file)
    
    test_trained_net(env, iterate=400, model_path="model_27000.pth")    

    env.close() 
