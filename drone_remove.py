import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'drone-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
