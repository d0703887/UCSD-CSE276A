from mpi_control import MegaPiController
import numpy as np
import time
import sys
import math
import heapq

from utils import kinematic_model, get_april_tag

class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f


class Robot:
    def __init__(self):
        self.mpi_control = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)
        self.fl = 30
        self.fr = 30
        self.br = 30
        self.bl = 30
        self.theta = 0
        self.target_frame = 0

        self.x_range = [-100, 132]
        self.y_range = [-82, 142]

        # self.x_range = [-110, 142]
        # self.y_range = [-92, 152]

        self.grid_size = 15

        #self.landmarks_pos = [(-12, -92), (80.5, -92), (140, -25.5), (140, 66.5), (32, 152), (48, 152), (140, 20), (81, 152), (-110, 129.5), (-110, -37), (-110, 74), (31.5, -92)]
        #self.mu = np.expand_dims(np.concatenate([np.zeros(3), np.array(self.landmarks_pos).flatten()]), 1)
        #self.sigma = np.diag([0 for _ in range(3 + 12 * 2)]).astype(np.float16)
        self.mu = np.zeros([3, 1], dtype=np.float16)
        self.sigma = np.diag([1, 1, 1]).astype(np.float16)

        self.num_landmarks = 0
        self.landmarks_id2index = {}
        self.index2landmarks_id = {}
        # self.landmarks_id2index = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11}
        # self.index2landmarks_id = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11}

    def is_valid(self, x, y):
        return not (x < self.x_range[0] + 25 or x > self.x_range[1] - 25 or y < self.y_range[0] + 25 or y > self.y_range[1] - 25)

    def update_boundaries(self):
        if self.num_landmarks > 0:
            x_coordinates = self.mu[3::2, 0]
            y_coordinates = self.mu[4::2, 0]
            if x_coordinates is not None:
                self.x_range[0] = np.min(x_coordinates)
                self.x_range[1] = np.max(x_coordinates)
            if y_coordinates is not None:
                self.y_range[0] = np.min(y_coordinates)
                self.y_range[1] = np.max(y_coordinates)

    # EKF-SLAM
    def compute_mu_hat(self, vx, vy, omega, dt):
        mu_hat = np.copy(self.mu)
        mu_hat[0][0] += vx * dt * math.cos(self.mu[2][0]) - vy * dt * math.sin(self.mu[2][0])
        mu_hat[1][0] += vx * dt * math.sin(self.mu[2][0]) + vy * dt * math.cos(self.mu[2][0])
        mu_hat[2][0] += omega * dt
        return mu_hat

    def compute_sigma_hat(self, vx, vy, omega, dt):
        m, _ = self.mu.shape
        Gt = np.identity(m)
        Gt[0][2] += -vx * dt * math.sin(self.mu[2][0]) - vy * dt * math.cos(self.mu[2][0])
        Gt[1][2] += vx * dt * math.cos(self.mu[2][0]) - vy * dt * math.sin(self.mu[2][0])

        Gt_T = Gt.transpose()
        ############
        alpha = 0.05  # 2% uncertainty
        beta = 0.05  # 2% uncertainty

        sigma_x = alpha * abs(vx) * dt
        sigma_y = alpha * abs(vy) * dt
        sigma_theta = beta * abs(omega) * dt

        R = np.diag([sigma_x ** 2, sigma_y ** 2, sigma_theta ** 2])
        ###########
        sigma_hat = Gt @ self.sigma @ Gt_T #  + R #TODO

        sigma_hat[:3, :3] += R
        return sigma_hat


    def initialize_landmark(self, landmark_id, r, phi, mu_hat, sigma_hat, Qt):
        self.num_landmarks += 1
        self.landmarks_id2index[landmark_id] = self.num_landmarks - 1
        self.index2landmarks_id[self.num_landmarks - 1] = landmark_id
        mu_hat = np.append(mu_hat, [[mu_hat[0][0] + r * math.cos(phi + mu_hat[2][0])], [mu_hat[1][0] + r * math.sin(phi + mu_hat[2][0])]], axis=0)

        new_sigma = np.zeros((3 + 2 * self.num_landmarks, 3 + 2 * self.num_landmarks))
        # new_sigma[:3 + 2 * (self.num_landmarks - 1), :3 + 2 * (self.num_landmarks - 1)] = sigma_hat
        # new_sigma[3 + 2 * (self.num_landmarks - 1)][3 + 2 * (self.num_landmarks - 1)] = 1
        # new_sigma[3 + 2 * (self.num_landmarks - 1) + 1][3 + 2 * (self.num_landmarks - 1) + 1] = 1

        landmark_jacobian = np.array([
            [math.cos(phi + mu_hat[2][0]), -r * math.sin(phi + mu_hat[2][0])],
            [math.sin(phi + mu_hat[2][0]), r * math.cos(phi + mu_hat[2][0])]
        ])
        Qt_global = landmark_jacobian @ Qt @ landmark_jacobian.T
        new_sigma[:3 + 2 * (self.num_landmarks - 1), :3 + 2 * (self.num_landmarks - 1)] = sigma_hat
        new_sigma[3 + 2 * (self.num_landmarks - 1):, 3 + 2 * (self.num_landmarks - 1):] = Qt_global
        sigma_hat = new_sigma

        return mu_hat, sigma_hat

    def get_Z(self):
        obs = get_april_tag()
        out = []
        for ob in obs:
            z = ob[1] + 7
            x = -(ob[2] - 2)
            if abs(ob[3] < 0.7):
                out.append(np.array([[ob[0]], [math.hypot(z, x)], [math.atan2(x, z)]]))
        return out

    def compute_z_hat(self, id, mu_hat):
        m, _ = mu_hat.shape
        delta_x = mu_hat[3 + 2 * self.landmarks_id2index[id]][0] - mu_hat[0][0]
        delta_y = mu_hat[3 + 2 * self.landmarks_id2index[id] + 1][0] - mu_hat[1][0]
        q = delta_x ** 2 + delta_y ** 2
        sqrt_q = math.sqrt(q)
        angle = math.atan2(delta_y, delta_x) - mu_hat[2][0]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return np.array([[sqrt_q], [angle]])

    def compute_Hti(self, id, mu_hat):
        m, _ = mu_hat.shape
        delta_x = mu_hat[3 + 2 * self.landmarks_id2index[id]][0] - mu_hat[0][0]
        delta_y = mu_hat[3 + 2 * self.landmarks_id2index[id] + 1][0] - mu_hat[1][0]
        q = delta_x ** 2 + delta_y ** 2
        sqrt_q = math.sqrt(q)
        low_Hi = np.array([[-sqrt_q * delta_x, -sqrt_q * delta_y, 0, sqrt_q * delta_x, sqrt_q * delta_y],
                           [delta_y, -delta_x, -q, -delta_y, delta_x]]) / q
        Fi = np.zeros((5, m))
        Fi[0][0] = 1
        Fi[1][1] = 1
        Fi[2][2] = 1
        Fi[3][3 + 2 * self.landmarks_id2index[id]] = 1
        Fi[4][3 + 2 * self.landmarks_id2index[id] + 1] = 1
        Hti = low_Hi @ Fi

        return Hti

    def slam_correction(self, mu_hat, sigma_hat):
        #print(f"Sigma_hat (diag): {np.diag(sigma_hat)}")

        Qt = np.diag([9, np.deg2rad(3) ** 2])
        Zs = self.get_Z()
        for z in Zs:
            landmark_id = z[0][0]

            if landmark_id not in self.landmarks_id2index:
                mu_hat, sigma_hat = self.initialize_landmark(landmark_id, z[1][0], z[2][0], mu_hat, sigma_hat, Qt)
                continue

            Hti = self.compute_Hti(landmark_id, mu_hat)
            z_hat = self.compute_z_hat(landmark_id, mu_hat)
            Kti = sigma_hat @ Hti.T @ np.linalg.inv((Hti @ sigma_hat @ Hti.T) + Qt)

            innovation = z[1:, :] - z_hat

            # Threshold for innovation
            if np.linalg.norm(innovation) > 50:  # Example threshold
                continue

            mu_hat = mu_hat + Kti @ innovation
            sigma_hat = (np.identity(mu_hat.shape[0]) - Kti @ Hti) @ sigma_hat


        mu_hat[2][0] = (mu_hat[2][0] + math.pi) % (2 * math.pi) - math.pi  # Normalize theta
        return mu_hat, sigma_hat


    def pretty_print(self, mu):
        print(f"Robot: {mu[0][0]}, {mu[1][0]}, {math.degrees(mu[2][0])}")
        out = []
        for idx in range(self.num_landmarks):
            out.append([self.index2landmarks_id[idx], mu[3 + 2 * idx][0], mu[3 + 2 * idx + 1][0]])
        for o in out:
            print(f"Landmark {int(o[0])}: ({o[1]}, {o[2]})")

    def step(self, vx, vy, omega, dt):
        mu_hat = self.compute_mu_hat(vx, vy, omega, dt)
        sigma_hat = self.compute_sigma_hat(vx, vy, omega, dt)
        self.mu, self.sigma = self.slam_correction(mu_hat, sigma_hat)
        #self.pretty_print(self.mu)
        # time.sleep(1)

    def coor2grid(self, x, y):
        return math.floor(x / self.grid_size), math.floor(y / self.grid_size)

    def grid_center(self, i, j):
        return self.grid_size * i + self.grid_size / 2, self.grid_size * j + self.grid_size / 2

    def main(self):
        open_pos = [[0, 0]]
        visited_pos = [[0, 0]]
        prev_pos = [0, 0]
        while not len(open_pos) == 0:
            np_open_pos = np.array(open_pos)
            x_dis = np_open_pos[:, 0] - prev_pos[0]
            y_dis = np_open_pos[:, 1] - prev_pos[1]
            dis = np.hypot(x_dis, y_dis)
            min_idx = np.argmin(dis)
            target_x, target_y = open_pos[min_idx]
            open_pos.pop(min_idx)

            print(f"Target: {target_x}, {target_y}")

            # rotation
            dx, dy = target_x - self.mu[0][0], target_y - self.mu[1][0]
            target_theta = math.atan2(dy, dx)
            while abs(self.mu[2][0] - target_theta) > 0.1:
                # wt, dt = self.mpi_control.rotate_with_angle(angle_diff)
                angle_diff = (target_theta - self.mu[2][0] + math.pi) % (2 * math.pi) - math.pi
                if angle_diff < 0:
                    self.mpi_control.carRotate(-43)
                    wt = -(43 * 0.022 * 1.19)
                else:
                    self.mpi_control.carRotate(43)
                    wt = (43 * 0.022 * 1.19)
                time.sleep(0.1)
                self.mpi_control.carStop()

                #time.sleep(1)
                self.step(0, 0, wt, 0.1)
                #print(f"{self.mu[0][0]}, {self.mu[1][0]}, {math.degrees(self.mu[2][0])}")


            # move to target point
            while not (target_x - 5 <= self.mu[0][0] <= target_x + 5 and target_y - 5 <= self.mu[1][0] <= target_y + 5):
                    dx, dy = target_x - self.mu[0][0], target_y - self.mu[1][0]
                    remain_distance = math.hypot(dx, dy)
                    target_theta = math.atan2(dy, dx)
                    # angle_diff = target_theta - self.mu[2][0]
                    # angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                    v_linear = 15
                    vx_world = v_linear * math.cos(target_theta)
                    vy_world = v_linear * math.sin(target_theta)
                    vx = vx_world * math.cos(self.mu[2][0]) + vy_world * math.sin(self.mu[2][0])
                    vy = -vx_world * math.sin(self.mu[2][0]) + vy_world * math.cos(self.mu[2][0])
                    m1, m2, m3, m4 = kinematic_model(vx=vx, vy=vy, omega=0)
                    self.mpi_control.setFourMotors(m1, m2, m3, m4)
                    dt = 0.1
                    time.sleep(dt)
                    self.mpi_control.carStop()
                    #time.sleep(1)
                    self.step(vx, vy, 0, dt)
                    #print(f"{self.mu[0][0]}, {self.mu[1][0]}, {math.degrees(self.mu[2][0])}")

            #self.pretty_print(self.mu)
            with open("tmp.txt", "a") as fp:
                fp.write(f"{self.mu[0][0]}, {self.mu[1][0]}, {math.degrees(self.mu[2][0])}\n")
            print(f"{self.mu[0][0]}, {self.mu[1][0]}, {math.degrees(self.mu[2][0])}")
            visited_pos.append([target_x, target_y])

            # Explore
            #print("Exploring")
            cur_grid = self.coor2grid(target_x, target_y)
            grid_candidate = [cur_grid + np.array([1, 0]),
                              cur_grid + np.array([0, 1]),
                              cur_grid + np.array([-1, 0]),
                              cur_grid + np.array([0, -1])]
            for grid in grid_candidate:
                x, y = self.grid_center(grid[0], grid[1])
                if self.is_valid(x, y) and [x, y] not in visited_pos and [x, y] not in open_pos:
                    open_pos.append([x, y])
            #print(open_pos)
            prev_pos = [target_x, target_y]

            # update boundaries
            #print("Updating boundaries")
            #self.update_boundaries()
            # print("x range", self.x_range)
            # print("y range", self.y_range)

            #print()


    def testing(self, target_x, target_y):
        self.mu[2][0] = math.pi / 2
        dx, dy = target_x - self.mu[0][0], target_y - self.mu[1][0]
        target_theta = math.atan2(dy, dx) % (2 * math.pi)
        while abs(self.mu[2][0] - target_theta) > 0.1:
            # wt, dt = self.mpi_control.rotate_with_angle(angle_diff)
            if target_theta - self.mu[2][0] < 0:
                self.mpi_control.carRotate(-43)
                wt = -(43 * 0.022 * 1.19)
            else:
                self.mpi_control.carRotate(43)
                wt = (43 * 0.022 * 1.19)
            time.sleep(0.1)
            self.mpi_control.carStop()

            time.sleep(2)
            self.step(0, 0, wt, 0.1)
            print(f"{self.mu[0][0]}, {self.mu[1][0]}, {math.degrees(self.mu[2][0])}")

# Run the test case
if __name__ == "__main__":
    r = Robot()
    r.main()
    #r.testing(-20, 0)



