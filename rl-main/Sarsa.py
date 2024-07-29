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


class SarsaPolicy(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256, device='cpu'):
        super(SarsaPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_value = nn.Linear(fc2_dims, n_actions)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_values = self.action_value(x)
        return action_values


class SarsaAgent:
    def __init__(self, input_dims, n_actions, gamma=0.99, lr=0.0003, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, device='cpu'):
        self.policy = SarsaPolicy(input_dims, n_actions, device=device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.random() > self.epsilon:
            action_values = self.policy(state)
            action = torch.argmax(action_values).item()
        else:
            action = np.random.choice(range(self.policy.action_value.out_features))
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        current_q = self.policy(state)[0, action]
        next_q = self.policy(next_state)[0, next_action]

        target = reward + self.gamma * next_q * (1 - int(done))
        loss = F.mse_loss(current_q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


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

    brain = SarsaAgent(
        input_dims=input_dims,
        n_actions=4,
        gamma=0.99,
        lr=0.0003,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device=device
    )

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

        state = [np.zeros(input_dims) for _ in range(len(all_junctions))]
        action = [0] * len(all_junctions)

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controled_lanes)
                total_time += waiting_time
                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controled_lanes)
                    state_ = list(vehicles_per_lane.values())

                    # Debug print to check the state dimensions
                    print(f"Junction {junction_number} state dimensions: {len(state_)} (expected: {input_dims})")

                    if len(state_) != input_dims:
                        print(f"Skipping junction {junction} due to mismatch in input dimensions.")
                        continue

                    reward = -1 * waiting_time
                    next_state = state_
                    next_action = brain.choose_action(next_state)
                    done = step == steps

                    if train:
                        brain.update(state[junction_number], action[junction_number], reward, next_state, next_action, done)
                        state[junction_number] = next_state
                        action[junction_number] = next_action

                    phaseDuration(junction, 6, select_lane[next_action][0])
                    phaseDuration(junction, min_duration + 10, select_lane[next_action][1])

                    if ard:
                        ph = str(traci.trafficlight.getPhase("0"))
                        value = write_read(ph)

                    traffic_lights_time[junction] = min_duration + 16
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
