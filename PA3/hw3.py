import numpy as np
import math
from utils import *
from mpi_control import MegaPiController

class ExtendedKalmanFilter:
    def __init__(self):
        # Initialize with assumption that no landmarks
        self.mpi_control = MegaPiController(port='/dev/ttyUSB0', verbose=True)
        time.sleep(1)
        self.mu = np.zeros((3, 1))
        self.sigma = np.zeros((3, 3))
        self.num_landmarks = 0
        self.landmarks_id2index = {}  # location of landmark i = (mu[3 + 2 * landmarks_idx2index[id]], mu[3 + 2 * landmarks_idx2index[id] + 1])
        self.index2landmarks_id = {}

    def compute_mu_hat(self, vx, vy, omega, dt):
        mu_hat = self.mu
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
        R = None # TODO

        sigma_hat = Gt @ self.sigma @ Gt_T # + R TODO
        return sigma_hat

    def initialize_landmark(self, landmark_id, r, phi, mu_hat, sigma_hat):
        self.num_landmarks += 1
        self.landmarks_id2index[landmark_id] = self.num_landmarks - 1
        self.index2landmarks_id[self.num_landmarks - 1] = landmark_id
        mu_hat = np.append(mu_hat, [[mu_hat[0][0] + r * math.cos(phi + mu_hat[2][0])], [mu_hat[1][0] + r * math.sin(phi + mu_hat[2][0])]], axis=0)
        print(f"Initialize landmark {landmark_id}: {mu_hat[-2][0]}, {mu_hat[-1][0]}")

        new_sigma = np.zeros((3 + 2 * self.num_landmarks, 3 + 2 * self.num_landmarks))
        new_sigma[:3 + 2 * (self.num_landmarks - 1), :3 + 2 * (self.num_landmarks - 1)] = sigma_hat
        new_sigma[3 + 2 * (self.num_landmarks - 1)][3 + 2 * (self.num_landmarks - 1)] = 1
        new_sigma[3 + 2 * (self.num_landmarks - 1) + 1][3 + 2 * (self.num_landmarks - 1) + 1] = 1
        sigma_hat = new_sigma
        return mu_hat, sigma_hat

    def get_Z(self):
        obs = get_april_tag()
        out = []
        for ob in obs:
            z = ob[1] + 7
            x = -(ob[2] - 2)
            out.append(np.array([[ob[0]], [math.hypot(z, x)], [math.atan(x / z)]]))
        return out

    def compute_z_hat(self, id, mu_hat):
        m, _ = mu_hat.shape
        delta_x = mu_hat[3 + 2 * self.landmarks_id2index[id]][0] - mu_hat[0][0]
        delta_y = mu_hat[3 + 2 * self.landmarks_id2index[id] + 1][0] - mu_hat[1][0]
        q = delta_x ** 2 + delta_y ** 2
        sqrt_q = math.sqrt(q)
        return np.array([[sqrt_q], [math.atan(delta_y / delta_x) - mu_hat[2][0]]])

    def compute_Hti(self, id, mu_hat):
        m, _ = mu_hat.shape
        delta_x = mu_hat[3 + 2 * self.landmarks_id2index[id]][0] - mu_hat[0][0]
        delta_y = mu_hat[3 + 2 * self.landmarks_id2index[id] + 1][0] - mu_hat[1][0]
        q = delta_x ** 2 + delta_y ** 2
        sqrt_q = math.sqrt(q)
        low_Hi = np.array([[-sqrt_q * delta_x, -sqrt_q * delta_y, 0, sqrt_q * delta_x, sqrt_q * delta_y], [delta_y, -delta_x, -q, -delta_y, delta_x]]) / q
        Fi = np.zeros((5, m))
        Fi[0][0] = 1
        Fi[1][1] = 1
        Fi[2][2] = 1
        Fi[3][3 + 2 * self.landmarks_id2index[id]] = 1
        Fi[4][3 + 2 * self.landmarks_id2index[id] + 1] = 1
        Hti = low_Hi @ Fi

        return Hti

    def slam_correction(self, mu_hat, sigma_hat):
        Qt = np.diag([0.01, np.deg2rad(1) ** 2])
        Zs = self.get_Z()
        for z in Zs:
            if abs(z[2][0]) > 0.4:
                continue
            print(f"Landmark{int(z[0][0])}: r={z[1][0]}, phi={z[2][0]}")

            landmark_id = z[0][0]
            if landmark_id not in self.landmarks_id2index:
                mu_hat, sigma_hat = self.initialize_landmark(landmark_id, z[1][0], z[2][0], mu_hat, sigma_hat)
            Hti = self.compute_Hti(z[0][0], mu_hat)
            z_hat = self.compute_z_hat(z[0][0], mu_hat)

            Kti = sigma_hat @ Hti.transpose() @ np.linalg.inv((Hti @ sigma_hat @ Hti.transpose()) + Qt)

            z = z[1:, :]  # remove frame_id
            mu_hat = mu_hat + Kti @ (z - z_hat)
            sigma_hat = (np.identity(mu_hat.shape[0]) - Kti @ Hti) @ sigma_hat
            #self.pretty_print(mu_hat)
        mu_hat[2][0] = (mu_hat[2][0] + math.pi) % (2 * math.pi) - math.pi  # normalize theta to [-pi, pi]
        return mu_hat, sigma_hat

    def pretty_print(self, mu):
        print(f"State\nRobot: {mu[0][0]}, {mu[1][0]}, {math.degrees(mu[2][0])}")
        out = []
        for idx in range(self.num_landmarks):
            out.append([self.index2landmarks_id[idx], mu[3 + 2 * idx][0], mu[3 + 2 * idx + 1][0]])
        for o in out:
            print(f"Landmark {int(o[0])}: ({o[1]}, {o[2]})")
        print()

    def step(self, vx, vy, omega, dt):
        mu_hat = self.compute_mu_hat(vx, vy, omega, dt)
        sigma_hat = self.compute_sigma_hat(vx, vy, omega, dt)

        self.mu, self.sigma = self.slam_correction(mu_hat, sigma_hat)
        self.pretty_print(self.mu)
        time.sleep(5)
    
    def move_to_point(self, target_point):
        print(target_point)
        target_x = target_point[0]
        target_y = target_point[1]
        dest_theta = target_point[2]

        dx = target_x - self.mu[0][0]
        dy = target_y - self.mu[1][0]
        remain_distance = math.hypot(dx, dy)
        target_theta = math.atan2(dy, dx)

        v_linear = 15
        vx_world = v_linear * math.cos(target_theta)
        vy_world = v_linear * math.sin(target_theta)
        vx = vx_world * math.cos(self.mu[2][0]) + vy_world * math.sin(self.mu[2][0])
        vy = -vx_world * math.sin(self.mu[2][0]) + vy_world * math.cos(self.mu[2][0])
        dt = remain_distance / v_linear
        m1, m2, m3, m4 = kinematic_model(vx=vx, vy=vy, omega=0)
        self.mpi_control.setFourMotors(m1, m2, m3, m4)
        time.sleep(dt)
        self.mpi_control.carStop()
        self.step(vx, vy, 0, dt)


        # while remain_distance > 8:
        #     v_linear = 15
        #     vx_world = v_linear * math.cos(target_theta)
        #     vy_world = v_linear * math.sin(target_theta)
        #     vx = vx_world * math.cos(self.mu[2][0]) + vy_world * math.sin(self.mu[2][0])
        #     vy = -vx_world * math.sin(self.mu[2][0]) + vy_world * math.cos(self.mu[2][0])
        #     m1, m2, m3, m4 = kinematic_model(vx=vx, vy=vy, omega=0)
        #     self.mpi_control.setFourMotors(m1, m2, m3, m4)
        #     time.sleep(0.5)
        #     self.mpi_control.carStop()
        #     self.step(vx, vy, 0, 0.5)
        #     dx = target_x - self.mu[0][0]
        #     dy = target_y - self.mu[1][0]
        #     remain_distance = math.hypot(dx, dy)
        #     target_theta = math.atan2(dy, dx)

        if self.mu[2] != dest_theta:
            angle_diff = (dest_theta - self.mu[2][0] - math.pi) % (2 * math.pi - math.pi)
            omega, dt = self.mpi_control.rotate_with_angle(angle_diff)
            self.step(0, 0, omega, dt)
            #self.mu[2][0] = dest_theta
        

def main():
    car = ExtendedKalmanFilter()
    waypoint = read_waypoint("waypoints.txt")
    for point in waypoint:
        car.move_to_point(point)
    print(car.sigma)

if __name__ == "__main__":
    main()