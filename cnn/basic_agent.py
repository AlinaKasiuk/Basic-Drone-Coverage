import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import SmoothL1Loss
import numpy as np
import matplotlib.pyplot as plt

from cnn.structure import DroneQNet
from constants import IMG_H, IMG_W


class BasicAgent:

    def __init__(self, actions, log_step=500, n_rows=5, sw: SummaryWriter = None):
        self.model = DroneQNet(2, IMG_W, IMG_H, len(actions))
        self.model.double()
        self.target_model = DroneQNet(2, IMG_W, IMG_H, len(actions))
        self.target_model.double()
        self.actions = actions

        self.criterion = SmoothL1Loss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        self.gamma = 0.8
        self.train_iterations = 0
        
        cuda = torch.cuda.is_available()
        self.device = 'cuda' if cuda else 'cpu'
        
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.log_step = log_step
        self.n_rows = n_rows
        self.log_folder = 'logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

        self.writer = sw

    def replace_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)

    def train(self, data):
        self.optimizer.zero_grad()

        current_states = np.array([i for i in data[:, 0]])
        next_states = np.array([i for i in data[:, 2]])
        model_input = torch.from_numpy(current_states)
        target_input = torch.from_numpy(next_states)
        actions_indexes = np.array(data[:, 1], dtype='int')
        rewards = np.array(data[:, 3], dtype='int')
        dones = np.array(data[:, 4], dtype='bool')

        y_hat = self.model(model_input.to(self.device))[np.arange(len(data)), actions_indexes]
        y_target = self.target_model(target_input.to(self.device)).max(dim=1)[0]

        y_target[dones] = 0.0
        target_q_value = torch.from_numpy(rewards).to(self.device) + self.gamma * y_target

        loss = self.criterion(target_q_value, y_hat)
        loss.backward()
        self.optimizer.step()

        self.train_iterations += 1

        if self.writer is not None:
            self.writer.add_scalar('loss', loss.item())

        if self.train_iterations % self.log_step == 0 and self.writer is not None:
            ndata = np.hstack([data, y_hat.view(-1, 1).cpu().detach().numpy(),
                               target_q_value.view(-1, 1).cpu().detach().numpy()])
            n_rows = min(self.n_rows, len(ndata))
            _, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(25, 10))

            ndata = np.random.permutation(ndata)[:n_rows]
            for i, d in enumerate(ndata):
                current_state = d[0]
                cs = np.concatenate((current_state[0], current_state[1]), axis=1)
                next_state = d[2]
                ns = np.concatenate((next_state[0], next_state[1]), axis=1)

                self.writer.add_image('current_state', cs, dataformats='HW', global_step=self.train_iterations)
                self.writer.add_image('next_state', ns, dataformats='HW', global_step=self.train_iterations)
                self.writer.add_scalar('selected action', d[1], self.train_iterations)
                self.writer.add_scalar('expected reward', d[3], self.train_iterations)
                self.writer.add_scalar('predicted q-value', d[-2], self.train_iterations)
                self.writer.add_scalar('target q-value', d[-1], self.train_iterations)

                axes[i, 0].imshow(cs)
                axes[i, 1].text(0.02, 0.5, str((self.actions[d[1]], d[3], d[-1], d[-2])))
                # axes[i, 1].set_title(str((self.actions[d[1]], d[3], d[-1], d[-2])))
                axes[i, 2].imshow(ns)
            # for ax in axes.ravel():
                # ax.axis('off')
            plt.savefig(os.path.join(self.log_folder, "log_{0}.png".format(self.train_iterations)))
