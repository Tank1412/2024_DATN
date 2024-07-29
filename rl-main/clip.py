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


class PPOPolicy(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, device='cpu'):
        super(PPOPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_head = nn.Linear(fc2_dims, n_actions)
        self.value_head = nn.Linear(fc2_dims, 1)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value


class PPOClipAgent:
    def __init__(self, input_dims, n_actions, gamma=0.99, lr=0.0003, betas=(0.9, 0.999), eps_clip=0.2, K_epochs=4, device='cpu'):
        self.policy = PPOPolicy(input_dims, n_actions, device=device)
        self.policy_old = PPOPolicy(input_dims, n_actions, device=device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.MseLoss = nn.MSELoss()
        self.device = device

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, _ = self.policy_old(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action), action_dist.entropy()

    def update(self, memory):
        for _ in range(self.K_epochs):
            states, actions, log_probs, rewards, dones = memory.sample()
            old_log_probs = log_probs.detach()
            rewards = rewards.detach()
            advantages = rewards - states.detach()

            for log_prob, reward, advantage in zip(old_log_probs, rewards, advantages):
                ratios = torch.exp(log_prob - old_log_probs)
                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(states, rewards)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float)
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        dones = torch.tensor(self.dones, dtype=torch.float)
        return states, actions, log_probs, rewards, dones

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []


def run(train=True, model_name="model", epochs=50, steps=500, ard=False, device='cpu'):
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

    brain = PPOClipAgent(
        input_dims=input_dims,
        n_actions=4,
        gamma=0.99,
        lr=0.0003,
        betas=(0.9, 0.999),
        eps_clip=0.2,
        K_epochs=4,
        device=device
    )

    memory = Memory()

    if not train:
        brain.policy.load_state_dict(torch.load(f'models/{model_name}.bin', map_location=device))

    print(brain.policy.device)
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
            prev_action[junction_number] = 0
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
                    done = step == steps

                    if train:
                        action, log_prob, _ = brain.choose_action(state_)
                        memory.add(state, action, log_prob, reward, done)
                        prev_action[junction_number] = action

                    phaseDuration(junction, 6, select_lane[prev_action[junction_number]][0])
                    phaseDuration(junction, min_duration + 10, select_lane[prev_action[junction_number]][1])

                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        value = write_read(ph)

                    traffic_lights_time[junction] = min_duration + 16
                    if done and train:
                        brain.update(memory)
                        memory.clear()
                else:
                    traffic_lights_time[junction] -= 1
            step += 1
        print("total_time", total_time)
        total_time_list.append(total_time)

        if total_time < best_time:
            best_time = total_time
            if train:
                torch.save(brain.policy.state_dict(), f'models/{model_name}.bin')

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(train=train, model_name=model_name, epochs=epochs, steps=steps, ard=ard, device=device)
