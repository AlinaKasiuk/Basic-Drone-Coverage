# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:00:39 2022

@author: Alina Kasiuk

Functions to output training results
"""
import os
from torch import save as torch_save
from sys import platform


def create_dir(name):
    "Creating the folders to save results"
    
    # Folder path define for different platforms. '\\', '/' issue
    if platform == "linux" or platform == "linux2":
        model_dir, tables_dir = _create_dir_linux(name)
    elif platform == "win32":
        model_dir, tables_dir = _create_dir_win32(name)
    else:
        print('unknown OS')
        
    if not os.path.exists(tables_dir):
         os.makedirs(tables_dir)
    if not os.path.exists(model_dir):
         os.makedirs(model_dir)
         
    return [model_dir, tables_dir]
    
def save_results(episode, path, model, table, table_actions):  
    "Saving the model and csv with training parameters"
    
    [model_path, tables_path] = path
    model_name=model_path+"model_{}.pth".format(episode+1)    
    _save_model(model, model_name)
    _save_tables(table, table_actions, tables_path)

def print_episode_info(episode, reward, timesteps, episode_dur, actions_dur):
    print("episode No", episode)
    print("Total reward:", reward)
    print("Episode finished after {0} timesteps".format(timesteps))
    print("Episode lasted {0:.2f} seconds".format(episode_dur)) 
    print("Avererage action duration {0:.3f} seconds".format(actions_dur/timesteps))   
    print('____________________________________________')
    
def write_info_file(episodes, repmem_limit, stoped=False):
    lines = ['Total number of episodes: {}'.format(episodes),
             'Replay memory limit: {}'.format(repmem_limit)]
    if stoped:
        lines.append('Stoped after {} episodes'.format(stoped))
        
    with open('info.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

def _create_dir_win32(name):
     tables_dir="tables\\{}\\".format(name)
     model_dir="models\\{}\\".format(name)         
     return model_dir, tables_dir
 
def _create_dir_linux(name):
     tables_dir="tables/{}/".format(name)
     model_dir="models/{}/".format(name)
     return model_dir, tables_dir 

def _save_model(model, path):
    torch_save(model.state_dict(), path)
    
def _save_tables(table, table_actions, path):
    table_actions.to_csv (path+"actions.csv", sep=';', index = False, header=True)
    table.to_csv (path+"episodes.csv", sep=';', index = False, header=True)
    

