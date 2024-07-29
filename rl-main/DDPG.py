from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for l in lanes:
        vehicle_per_lane[l] = 0
        for k in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[l] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time


def phaseDuration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))
        return actions

class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class Agent:
    def __init__(
        self,
        gamma,
        tau,
        lr_actor,
        lr_critic,
        input_dims,
        fc1_dims,
        fc2_dims,
        batch_size,
        n_actions,
        max_memory_size=100000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.memory_cntr = 0
        self.max_mem = max_memory_size
        self.device = device

        self.actor = Actor(self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions).to(self.device)
        self.critic = Critic(self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions).to(self.device)
        self.target_actor = Actor(self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions).to(self.device)
        self.target_critic = Critic(self.input_dims, self.fc1_dims, self.fc2_dims, self.n_actions).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.loss = nn.MSELoss()

        self.memory = {
            "state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
            "new_state_memory": np.zeros((self.max_mem, self.input_dims), dtype=np.float32),
            "reward_memory": np.zeros(self.max_mem, dtype=np.float32),
            "action_memory": np.zeros((self.max_mem, self.n_actions), dtype=np.float32),
            "terminal_memory": np.zeros(self.max_mem, dtype=np.bool_),
        }

        self.update_network_parameters(tau=1)

    def store_transition(self, state, state_, action, reward, done):
        index = self.memory_cntr % self.max_mem
        self.memory["state_memory"][index] = state
        self.memory["new_state_memory"][index] = state_
        self.memory["reward_memory"][index] = reward
        self.memory["action_memory"][index] = action
        self.memory["terminal_memory"][index] = done
        self.memory_cntr += 1

    def choose_action(self, observation):
        self.actor.eval()
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        self.actor.train()
        return action

    def learn(self):
        if self.memory_cntr < self.batch_size:
            return

        max_mem = min(self.memory_cntr, self.max_mem)
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.memory["state_memory"][batch_indices]).to(self.device)
        new_state_batch = torch.tensor(self.memory["new_state_memory"][batch_indices]).to(self.device)
        reward_batch = torch.tensor(self.memory["reward_memory"][batch_indices]).to(self.device)
        terminal_batch = torch.tensor(self.memory["terminal_memory"][batch_indices]).to(self.device)
        action_batch = torch.tensor(self.memory["action_memory"][batch_indices]).to(self.device)

        target_actions = self.target_actor(new_state_batch)
        critic_value_ = self.target_critic(new_state_batch, target_actions)
        critic_value = self.critic(state_batch, action_batch)

        critic_value_[terminal_batch] = 0.0
        target = reward_batch + self.gamma * critic_value_

        self.critic_optimizer.zero_grad()
        critic_loss = self.loss(target, critic_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.target_actor.state_dict(), 'target_actor.pth')
        torch.save(self.critic.state_dict(), 'critic.pth')
        torch.save(self.target_critic.state_dict(), 'target_critic.pth')

    def load_models(self):
        self.actor.load_state_dict(torch.load('actor.pth'))
        self.target_actor.load_state_dict(torch.load('target_actor.pth'))
        self.critic.load_state_dict(torch.load('critic.pth'))
        self.target_critic.load_state_dict(torch.load('target_critic.pth'))


def run(train=True, model_name="model", epochs=50, steps=500, ard=False):
    if ard:
        arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
        def write_read(x):
            arduino.write(bytes(x, 'utf-8'))
            time.sleep(0.05)
            data = arduino.readline()
            return data

    epochs = epochs
    steps = steps
    best_time = np.inf
    total_time_list = list()
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"]
    )
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    input_dims = len(traci.trafficlight.getControlledLanes(all_junctions[0]))  # Adjust input dimensions dynamically

    brain = Agent(
        gamma=0.99,
        tau=0.001,
        lr_actor=0.0001,
        lr_critic=0.001,
        input_dims=input_dims,
        fc1_dims=256,
        fc2_dims=256,
        batch_size=64,
        n_actions=4,
    )

    if not train:
        brain.load_models()

    print(brain.device)
    traci.close()
    for e in range(epochs):
        if train:
            traci.start(
                [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )
        else:
            traci.start(
                [checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"]
            )

        print(f"epoch: {e}")
        select_lane = [
            ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],
            ["rrryyyrrrrrr", "rrrGGGrrrrrr"],
            ["rrrrrryyyrrr", "rrrrrrGGGrrr"],
            ["rrrrrrrrryyy", "rrrrrrrrrGGG"],
        ]

        step = 0
        total_time = 0
        min_duration = 5

        traffic_lights_time = dict()
        prev_wait_time = dict()
        prev_vehicles_per_lane = dict()
        prev_action = dict()
        all_lanes = list()

        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = np.zeros(brain.n_actions)
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * input_dims
            all_lanes.extend(list(traci.trafficlight.getControlledLanes(junction)))

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controled_lanes)
                total_time += waiting_time
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    state_ = list(vehicles_per_lane.values())

                    if len(state_) != input_dims:
                        print(f"Skipping junction {junction} due to mismatch in input dimensions.")
                        continue

                    reward = -1 * waiting_time
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number], reward, (step == steps))

                    action = brain.choose_action(state_)
                    prev_action[junction_number] = action
                    phaseDuration(junction, 6, select_lane[int(action[0])][0])
                    phaseDuration(junction, min_duration + 10, select_lane[int(action[0])][1])

                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        value = write_read(ph)

                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn()
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total_time", total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                brain.save_models()

        traci.close()
        sys.stdout.flush()
        if not train:
            break
    if train:
        plt.plot(list(range(len(total_time_list))), total_time_list)
        plt.xlabel("epochs")
        plt.ylabel("total time")
        plt.savefig(f'plots/time_vs_epoch_{model_name}.png')
        plt.show()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m",
        dest='model_name',
        type='string',
        default="model",
        help="name of model",
    )
    optParser.add_option(
        "--train",
        action='store_true',
        default=False,
        help="training or testing",
    )
    optParser.add_option(
        "-e",
        dest='epochs',
        type='int',
        default=50,
        help="Number of epochs",
    )
    optParser.add_option(
        "-s",
        dest='steps',
        type='int',
        default=500,
        help="Number of steps",
    )
    optParser.add_option(
       "--ard",
        action='store_true',
        default=False,
        help="Connect Arduino",
    )
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()
    model_name = options.model_name
    train = options.train
    epochs = options.epochs
    steps = options.steps
    ard = options.ard
    run(train=train, model_name=model_name, epochs=epochs, steps=steps, ard=ard)
