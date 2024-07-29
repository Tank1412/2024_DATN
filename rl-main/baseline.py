from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure SUMO_HOME environment variable is set
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

def run_baseline(steps=500):
    # Start SUMO simulation
    traci.start(
        [checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "baseline_tripinfo.xml"]
    )

    all_junctions = traci.trafficlight.getIDList()
    step = 0
    total_time = 0

    while step <= steps:
        traci.simulationStep()
        for junction in all_junctions:
            controled_lanes = traci.trafficlight.getControlledLanes(junction)
            waiting_time = get_waiting_time(controled_lanes)
            total_time += waiting_time
        step += 1

    print("Baseline total_time:", total_time)
    traci.close()

if __name__ == "__main__":
    run_baseline(steps=500)
