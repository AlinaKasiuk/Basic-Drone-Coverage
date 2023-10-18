from torch import load as torch_load
from sys import platform

from cnn.structure import DroneQNet

from constants import IMG_H, IMG_W, actions
from train_RL import get_current_state, select_action, init_environment

if platform == "win32":
    # Remove drone-v0 from registry
    import drone_remove
    drone_remove
    
from gym_drone.envs.drone_env import DroneEnv
from cnn.basic_agent import BasicAgent


def load_model(path):
    model = DroneQNet(2, IMG_W, IMG_H, len(actions))
    model.load_state_dict(torch_load(path))
    model.eval()
    model.double()
    return model


def test_trained_net(env, iterate=50, model_path="drone_model.pth"):
    agent = BasicAgent(actions)
    model = load_model(model_path)
    model.to(agent.device)

    state_matrix, cameraspot = env.reset()
    cs = get_current_state(env,state_matrix, cameraspot)

    for i in range(iterate):
        env.render(show=True)
               #    i > 190)
        a, a_type = select_action(model, cs, 0, agent.device)
        observation, reward, done, _ = env.step(a)

        state_matrix, _, cameraspot = observation
        cs = get_current_state(env,state_matrix, cameraspot)


if __name__ == '__main__':
    
    m_file = "ones.csv"
    env = init_environment(map_file=m_file)
    test_trained_net(env, iterate=400, model_path="model_10000.pth")
    env.close() 
