"""
This file include code that control the robot motors
"""
from megapi import MegaPi
import sys
import math
import numpy as np
from utils import interpolate_motor_value, Wheel_Radius, kinematic_model, get_april_tag
import time
MFR = 2     # port for motor front right
MBL = 3     # port for motor back left
MBR = 10    # port for motor back right
MFL = 11  # port for motor front left
angular_speed = 0.022
file_path = "./waypoints.txt"



class MegaPiController:
    def __init__(self, port='/dev/ttyUSB0', verbose=True):
        self.port = port
        self.verbose = verbose
        if verbose:
            self.printConfiguration()
        self.bot = MegaPi()
        self.bot.start(port=port)
        self.mfr = MFR  # port for motor front right
        self.mbl = MBL  # port for motor back left
        self.mbr = MBR  # port for motor back right
        self.mfl = MFL  # port for motor front left

        # customized variable
        # x, y, theta
        self.coordinate = np.array([0, 0, 0], dtype=float)
        self.angular_speed = angular_speed
        self.kv = 1.18
        self.kh = 0.92
        self.distance_to_frame = 50

    def printConfiguration(self):
        print('MegaPiController:')
        print("Communication Port:" + repr(self.port))
        print("Motor ports: MFR: " + repr(MFR) +
              " MBL: " + repr(MBL) +
              " MBR: " + repr(MBR) +
              " MFL: " + repr(MFL))

    def setFourMotors(self, vfl=0, vfr=0, vbl=0, vbr=0):
        if self.verbose:
            print("Set Motors: vfl: " + repr(int(round(vfl, 0))) +
                  " vfr: " + repr(int(round(vfr, 0))) +
                  " vbl: " + repr(int(round(vbl, 0))) +
                  " vbr: " + repr(int(round(vbr, 0))))
        self.bot.motorRun(self.mfl, round(-vfl))
        self.bot.motorRun(self.mfr, round(vfr))
        self.bot.motorRun(self.mbl, round(-vbl))
        self.bot.motorRun(self.mbr, round(vbr))

    def carStop(self):
        if self.verbose:
            self.bot.motorRun(self.mfl, 0)
            self.bot.motorRun(self.mfr, 0)
            self.bot.motorRun(self.mbl, 0)
            self.bot.motorRun(self.mbr, 0)
            # print("CAR STOP")

    def carStraight(self, speed):
        # if self.verbose:
        #     print("CAR STRAIGHT:")
        self.setFourMotors(speed, speed, speed, speed)

    def carRotate(self, speed):
        # if self.verbose:
        #     print("CAR ROTATE:")
        self.setFourMotors(-speed, speed, -speed, speed)

    def rotate_with_angle(self, angle):
        print(f"Rotate {math.degrees(angle)} degrees")
        t = abs(angle) / (43 * self.angular_speed * self.kh)
        if angle < 0:
            self.carRotate(-43)
        else:
            self.carRotate(43)
        time.sleep(t)
        self.carStop()

    def run_straight(self, distance):
        print(f"Run straight for {distance}")
        angular_velocity = 7.018
        t = (abs(distance) * self.kv) / (angular_velocity * Wheel_Radius)
        m1, m2, m3, m4 = interpolate_motor_value([angular_velocity, angular_velocity, angular_velocity, angular_velocity])

        if distance < 0:
            self.setFourMotors(-m1, -m2, -m3, -m4)
        else:
            self.setFourMotors(m1, m2, m3, m4)
        time.sleep(t)
        self.carStop()

    def slide_(self):
        return

    def point_to_point(self, target_point):
        dx, dy, _ = target_point - self.coordinate
        target_theta = math.atan2(dy, dx)
        angle_difference = target_theta - self.coordinate[2]
        distance = math.hypot(dx, dy)

        if abs(angle_difference) > math.pi / 2:
            angle_difference = -(math.pi - abs(angle_difference)) if angle_difference > 0 else (math.pi - abs(angle_difference))
            distance = -distance

        self.coordinate[2] += angle_difference
        # rotate toward target point
        self.rotate_with_angle(angle_difference)
        # run toward target point
        self.run_straight(distance)
        # rotate to designated angle
        self.rotate_with_angle(target_point[2] - self.coordinate[2])

        self.coordinate = target_point

    # hw2
    def align_with_frame(self, frame, target_theta):
        print(f"Aligning to apriltag {frame}")
        while True:
            time.sleep(1)
            frame_id, pos_z, pos_x, orient_y = get_april_tag(frame)
            print(f"orientation y: {orient_y}")

            if abs(orient_y) > 0.06:
                self.coordinate[2] = target_theta + orient_y / 2
                self.rotate_with_angle(-orient_y / 2)
            else:
                break
        self.carStop()
        print(f"Finish aligning to apriltag {frame}\n")

    def move_to_frame(self, frame, target_point, f2w_rotation, f2w_translation):
        print(f"Moving to apriltag {frame}")
        while True:
            time.sleep(1)
            frame_id, pos_z, pos_x, orient_y = get_april_tag(frame)
            print(f"position z: {pos_z}")
            print(f"position x: {pos_x}")
            print(f"orientation y: {orient_y}")

            total_z = pos_z + pos_x * math.tan(orient_y) + 8
            c_r_f = np.array([-pos_x / math.cos(orient_y) + total_z * math.sin(orient_y), -total_z * math.cos(orient_y)])

            c_r_w = np.matmul(np.linalg.inv(f2w_rotation), (c_r_f - f2w_translation))
            self.coordinate[0] = c_r_w[0]
            self.coordinate[1] = c_r_w[1]
            print(f"Robot coordinate: ({self.coordinate[0]}, {self.coordinate[1]})")

            dx, dy, _ = target_point - self.coordinate
            target_theta = math.atan2(dy, dx)
            remain_distance = math.hypot(dx, dy)
            if remain_distance > 5:
                v_linear = 23
                vx_world = v_linear * math.cos(target_theta)
                vy_world = v_linear * math.sin(target_theta)
                vx = vx_world * math.cos(self.coordinate[2]) + vy_world * math.sin(self.coordinate[2])
                vy = -vx_world * math.sin(self.coordinate[2]) + vy_world * math.cos(self.coordinate[2])
                print(f"vx: {vx}, vy: {vy}\n")
                m1, m2, m3, m4 = kinematic_model(vx=vx, vy=vy, omega=0)
                self.setFourMotors(m1, m2, m3, m4)
                time.sleep(0.3)
                self.carStop()
            else:
                break
        print(f"Finish moving to apriltag {frame}\n")

    def close_loop_control(self, target_point, target_frame, align_frame, f2w_rotation, f2w_translation):
        dx, dy, _ = target_point - self.coordinate
        target_theta = math.atan2(dy, dx)
        angle_difference = target_theta - self.coordinate[2]
        angle_difference = (angle_difference + 2 * math.pi) % (2 * math.pi)

        # Rotate roughly to target theta
        self.rotate_with_angle(angle_difference)
        self.coordinate[2] = (self.coordinate[2] + angle_difference) % (2 * math.pi)

        # Rotate until align with new target april tag
        self.align_with_frame(target_frame, target_theta)
        print(f"Robot theta: {self.coordinate[2]}\n")

        # Move to target tag
        self.move_to_frame(target_frame, target_point, f2w_rotation, f2w_translation)
        print(f"Robot coordinate: ({self.coordinate[0]}, {self.coordinate[1]})\n")

        # Rotate roughly to designated theta
        angle_difference = target_point[2] - self.coordinate[2]
        self.rotate_with_angle(angle_difference)
        self.coordinate[2] = (self.coordinate[2] + angle_difference) % (2 * math.pi)

        # Rotate until align with align april tag
        self.align_with_frame(align_frame, target_point[2])
        print(f"Robot theta: {self.coordinate[2]}\n")

if __name__ == "__main__":
    exit(0)
