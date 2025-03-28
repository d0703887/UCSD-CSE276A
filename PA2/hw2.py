import numpy as np
import time
from mpi_control import MegaPiController
import math
from utils import kinematic_model, get_april_tag, read_waypoint


def main():
    mpi_control = MegaPiController(port='/dev/ttyUSB0', verbose=True)
    time.sleep(1)

    waypoints = read_waypoint("waypoints.txt")
    target_frame = [0, 1, 3]
    align_frame = [0, 2, 0]
    f2w_rotation = np.array(
        ([[0, -1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [-1, 0]], [[-0.894429, 0.447209], [-0.447209, -0.894429]]))
    f2w_translation = np.array(([0, -100], [-50, -150], [-100, 0], [0, -50]))

    for i, target_point in enumerate(waypoints):
        print(target_point)
        mpi_control.close_loop_control(target_point, target_frame[i], align_frame[i], f2w_rotation[target_frame[i]], f2w_translation[target_frame[i]])


if __name__ == "__main__":
    main()
