import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import time
import os

Lx = 6.9
Ly = 5.65
Wheel_Radius = 2.5
Kinematic_Matrix = np.array([[1, -1, -(Lx + Ly)], [1, 1, Lx + Ly], [1, 1, -(Lx + Ly)], [1, -1, Lx + Ly]])


def adjust_vy(vy: float):
    vy = 1.16279 * vy + 9.76744
    return vy


def interpolate_motor_value(angular_speeds):
    # Calibration data as dictionaries for each wheel
    # 0: front left, 1: front right, 2: back left, 3: back right
    calibration_data = [
        [(3.307, 30), (5.108280738, 40), (6, 45), (6.981317008, 50), (7.6, 55)],
        [(4.05366794, 30), (5.872135801, 40), (6.82, 45),  (7.623, 50)],
        [(4.05366794, 30), (5.872135801, 40), (6.82, 45),  (7.623, 50)],
        [(4.05366794, 30), (5.872135801, 40), (6.82, 45),  (7.623, 50)]
    ]
    motor_values = []
    for i in range(4):
        x_data, y_data = zip(*calibration_data[i])
        coefficients = np.polyfit(x_data, y_data, 1)
        a, b = coefficients
        motor_values.append(a * abs(angular_speeds[i]) + b if angular_speeds[i] > 0 else -(a * abs(angular_speeds[i]) + b))

    return motor_values


def kinematic_model(vx, vy, omega):
    vy = adjust_vy(vy)
    wheel_speeds = np.matmul(Kinematic_Matrix, np.array([[vx], [vy], [omega]]))
    angular_speeds = wheel_speeds / Wheel_Radius
    m1, m2, m3, m4 = interpolate_motor_value(angular_speeds)
    return m1.item(), m2.item(), m3.item(), m4.item()


class aprilDetectionNode(Node):
    def __init__(self):
        super().__init__("aprilDetectionNode")
        self.sub = self.create_subscription(PoseStamped, "/april_poses", self.listener_callback, 10)

    def listener_callback(self, msg):
        with open("pose_estimation.txt", "w") as fp:
            fp.writelines([msg.header.frame_id + "\n", str(msg.pose.position.x) + "\n", str(msg.pose.position.z)+ "\n", str(msg.pose.orientation.y)+ "\n"])


# read target april tag poses
def get_april_tag(tag_id: int):
    rclpy.init()
    node = aprilDetectionNode()

    avg = np.zeros(4)
    count = 0
    for i in range(10):
        rclpy.spin_once(node)

        with open("pose_estimation.txt", "r") as fp:
            data = fp.read()

        if int(data[0]) == tag_id:
            count += 1
            avg += np.array(data.split(), dtype=float)

    avg /= count
    rclpy.shutdown()
    return avg[0], avg[2] * 100, avg[1] * 100, avg[3]


def read_waypoint(file_path):
    waypoints = []
    with open(file_path, "r") as fp:
        text = fp.read().split("\n")
        for t in text[1:]:
            target_x, target_y, target_theta = t.split(",")
            target_x = int(target_x) / 2 * 100
            target_y = int(target_y) / 2 * 100
            waypoints.append(np.array([target_x, target_y, float(target_theta)]))
    return waypoints



