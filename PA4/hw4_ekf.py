from mpi_control import MegaPiController
import numpy as np
import time
import sys
import math
import heapq

from utils import kinematic_model, get_april_tag, read_waypoint

file_path = "./waypoints.txt"


class Block:
    def __init__(self):
        self.x1 = -91
        self.x2 = -152
        self.y1 = 104
        self.y2 = 165

    def get_four_points(self):
        return ((self.x1, self.y1), (self.x1, self.y2), (self.x2, self.y1), (self.x2, self.y2))


class Node:
    def __init__(self, x, y):
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

        self.x_range = [-244, 0]
        self.y_range = [0, 244]

        mu = [-38, 38, 0, 0, 91.44, 0, 182.8, -91.4, 244, -182.8, 244, -244, 91.44, -244, 182.8, -91.4, 0, -182.8, 0]
        self.mu = np.array(mu).reshape(-1, 1)

        sigma = np.zeros((19, 19))
        for i in range(3, 19):
            sigma[i][i] = 1
        self.sigma = sigma

        self.landmarks = {
            0: np.array([0, 91.44]),
            1: np.array([0, 182.8]),
            2: np.array([-91.4, 244]),
            3: np.array([-182.8, 244]),
            4: np.array([-244, 91.44]),
            5: np.array([-244, 182.8]),
            7: np.array([-91.4, 0]),
            8: np.array([-182.8, 0]),
        }
        self.num_landmarks = 0
        self.landmarks_id2index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7}

        self.f2w_rotation = np.array([[[0, -1],
                                       [1, 0]],

                                      [[0, -1],
                                       [1, 0]],

                                      [[1, 0],
                                       [0, 1]],

                                      [[1, 0],
                                       [0, 1]],

                                      [[0, 1],
                                       [-1, 0]],

                                      [[0, 1],
                                       [-1, 0]],

                                      [[-1, 0],
                                       [-0, -1]],

                                      [[-1, 0],
                                       [0, -1]]], dtype=float)

        self.f2w_translation = [[91.44, 0.],
                                [182.88, 0.],
                                [91.44, -243.84],
                                [182.88, -243.84],
                                [-91.44, -243.84],
                                [-182.88, -243.84],
                                [-91.44, 0.],
                                [-182.44, 0.]]

    def a_star(self, target_point: Node, block: Block, mode: str = "fast"):
        robot_step = 5
        start_node = Node(self.mu[0][0], self.mu[1][0])
        open_list = []
        closed_set = set()
        open_set = set()
        open_set.add((start_node.x, start_node.y))
        open_list.append((start_node.f, start_node))
        heapq.heapify(open_list)

        while open_list:
            f, curr_node = heapq.heappop(open_list)
            open_set.remove((curr_node.x, curr_node.y))  # Remove the node from the open set
            closed_set.add((curr_node.x, curr_node.y))  # Add to closed set
            print(f"Processing node: ({curr_node.x}, {curr_node.y}) with f: {curr_node.f}")
            # closed_list.append((curr_node.x,curr_node.y))
            steps = [(robot_step, 0), (-robot_step, 0), (0, -robot_step), (0, robot_step), (robot_step, robot_step),
                     (-robot_step, robot_step), (-robot_step, robot_step), (robot_step, -robot_step)]
            for step_x, step_y in steps:
                new_x = curr_node.x + step_x
                new_y = curr_node.y + step_y

                if self.is_valid(new_x, new_y) and self.is_unblocked(new_x, new_y, block) and (
                        new_x, new_y) not in closed_set:
                    print(f"Checking node: ({new_x}, {new_y})")
                    succ = Node(new_x, new_y)

                    # If the successor is the destination
                    if self.reach_goal(new_x, new_y, target_point.x, target_point.y):
                        # Set the parent of the destination cell
                        succ.parent = curr_node
                        print("The destination cell is found")
                        # Trace and print the path from source to destination
                        return self.trace_path(succ)
                    else:
                        succ.g = curr_node.g + robot_step
                        h = self.calculate_h(new_x, new_y, target_point.x, target_point.y, block, mode)
                        succ.h = h
                        succ.f = succ.g + succ.h
                        succ.parent = curr_node
                        # print(f" Successor node: ({succ.x}, {succ.y}) with f: {succ.f}, g: {succ.g}, h: {succ.h}")

                        # inOpen = False
                        # for f, node in open_list:
                        # if (new_x, new_y) == node:
                        # if succ.f > f:
                        # print(f" Node ({succ.x}, {succ.y}) already in open list with better f.")
                        # inOpen = True
                        # if not inOpen:
                        # heapq.heappush(open_list, (succ.f, succ))

                        if (new_x, new_y) not in open_set:
                            heapq.heappush(open_list, (succ.f, succ))
                            open_set.add((new_x, new_y))  # Add to open set to avoid duplicates
                            # print(f" Node ({succ.x}, {succ.y}) added to open list.")
        # print("Failed to find a path to the destination.")

    def is_valid(self, x, y):
        return not (x < self.x_range[0] or x > self.x_range[1] or y < self.y_range[0] or y > self.y_range[1])

    def reach_goal(self, x, y, target_x, target_y):
        return x in range(target_x - 5, target_x + 5) and y in range(target_y - 5, target_y + 5)

    def calculate_h(self, x, y, target_x, target_y, block: Block, mode: str):
        h = float("inf")
        x1, x2, y1, y2 = block.x1, block.x2, block.y1, block.y2
        if (x > x1 and x < x2) or (y > y1 and y < y2):
            if x > x1 and x < x2:
                h = min(h, min(abs(y - y1), abs(y - y2)))

            if y > y1 and y < y2:
                h = min(h, min(abs(x - x1), abs(x - x2)))
        else:
            dis = np.array(block.get_four_points())
            dis[:, 0] -= x
            dis[:, 1] -= y
            dis = np.abs(dis)

            dis = np.min(np.hypot(dis[:, 0], dis[:, 1]))
            h = dis
        # print(f"x={x}, y={y}, h={h}")
        h = math.hypot(abs(target_x - x), abs(target_y - y)) + (
            500 if mode == "safe" else 100) / h if h != 0 else math.hypot(abs(target_x - x), abs(target_y - y)) + 1000
        return h

    def trace_path(self, node):
        path = []
        current = node

        # Backtrack from the destination node to the start node
        while current:
            # print(current.x, current.y)
            path.append((current.x, current.y))
            if current.parent.x == self.mu[0][0] and current.parent.y == self.mu[1][0]:  # Reached the start node
                path.append((current.parent.x, current.parent.y))
                break
            current = current.parent  # Move to the parent node

        # The path will be in reverse order, so reverse it to get the correct order
        path.reverse()

        return path

    def is_unblocked(self, x, y, block):
        if min(block.x1, block.x2) <= x <= max(block.x1, block.x2) and min(block.y1, block.y2) <= y <= max(block.y1,
                                                                                                           block.y2):
            return False
        return True

    '''
    def main(self, end_x, end_y, block, mode):
        path = self.a_star(Node(end_x, end_y), block, mode)

        for point in path:
            target_x, target_y = point
            print(f"Target Point: ({target_x / 2.54}, {target_y / 2.54})")
            dx, dy = target_x - self.mu[0][0], target_y - self.mu[1][0]
            remain_distance = math.hypot(dx, dy)
            target_theta = math.atan2(dy, dx)
            v_linear = 23
            vx_world = v_linear * math.cos(target_theta)
            vy_world = v_linear * math.sin(target_theta)
            vx = vx_world * math.cos(self.theta) + vy_world * math.sin(self.theta)
            vy = -vx_world * math.sin(self.theta) + vy_world * math.cos(self.theta)
            m1, m2, m3, m4 = kinematic_model(vx=vx, vy=vy, omega=0)
            self.mpi_control.setFourMotors(m1, m2, m3, m4)
            time.sleep(remain_distance / v_linear)
            self.mpi_control.carStop()

            self.mu[0][0], self.mu[1][0] = target_x, target_y
            time.sleep(10)
            self.correction()
            print()
    '''

    '''
    def correction(self):
        obs = get_april_tag()
        out = []

        for ob in obs:
            id = int(ob[0])
            z = ob[1] + 7
            x = -(ob[2] - 2)
            orient_y = ob[3]
            total_z = z - x * math.tan(orient_y)
            c_r_f = np.array([x / math.cos(orient_y) + total_z * math.sin(orient_y), -z * math.cos(orient_y)])
            theta_r_f = orient_y + math.pi / 2
            print(c_r_f / 2.54)
            c_r_w = np.matmul(np.linalg.inv(self.f2w_rotation[id]), (c_r_f - self.f2w_translation[id]))
            out.append([c_r_w[0], c_r_w[1], theta_r_f - math.acos(self.f2w_rotation[id][0][0])])

        if len(out) != 0:
            out = np.array(out)
            out = np.mean(out, axis=0)
            print(f'Estimated Robot Coordinate: ({out[0] / 2.54}, {out[1] / 2.54}, {math.degrees(out[2])})')
            print(f'Current Robot Coordinate: ({self.x / 2.54}, {self.y / 2.54}, {math.degrees(self.theta)})')
            self.x, self.y, self.theta = out

        print(f"Robot Coordinate: ({self.x / 2.54}, {self.y / 2.54}, {math.degrees(self.theta)})")
    '''
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
        alpha = 0.1  # 10% uncertainty
        beta = 0.1  # 10% uncertainty

        sigma_x = alpha * abs(vx) * dt
        sigma_y = alpha * abs(vy) * dt
        sigma_theta = beta * abs(omega) * dt

        R = np.diag([sigma_x ** 2, sigma_y ** 2, sigma_theta ** 2])
        ###########
        sigma_hat = Gt @ self.sigma @ Gt_T #  + R #TODO

        sigma_hat[:3, :3] += R
        return sigma_hat

    def get_Z(self):
        obs = get_april_tag()
        out = []
        for ob in obs:
            z = ob[1] + 7
            x = -(ob[2] - 2)
            out.append(np.array([[ob[0]], [math.hypot(z, x)], [math.atan2(x, z)]]))
        return out

    def compute_z_hat(self, id, mu_hat):
        m, _ = mu_hat.shape
        delta_x = mu_hat[3 + 2 * self.landmarks_id2index[id]][0] - mu_hat[0][0]
        delta_y = mu_hat[3 + 2 * self.landmarks_id2index[id] + 1][0] - mu_hat[1][0]
        q = delta_x ** 2 + delta_y ** 2
        sqrt_q = math.sqrt(q)
        return np.array([[sqrt_q], [math.atan2(delta_y, delta_x) - mu_hat[2][0]]])

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
        Qt = np.diag([0.01, np.deg2rad(1) ** 2])
        Zs = self.get_Z()
        for z in Zs:
            if abs(z[2][0]) > 0.5:
                continue
            print(f"Landmark{int(z[0][0])}: r={z[1][0]}, phi={z[2][0]}")

            landmark_id = z[0][0]
            Hti = self.compute_Hti(z[0][0], mu_hat)
            z_hat = self.compute_z_hat(z[0][0], mu_hat)

            Kti = sigma_hat @ Hti.transpose() @ np.linalg.inv((Hti @ sigma_hat @ Hti.transpose()) + Qt)

            z = z[1:, :]  # remove frame_id
            mu_hat = mu_hat + Kti @ (z - z_hat)
            sigma_hat = (np.identity(mu_hat.shape[0]) - Kti @ Hti) @ sigma_hat
            # self.pretty_print(mu_hat)
        mu_hat[2][0] = (mu_hat[2][0] + math.pi) % (2 * math.pi) - math.pi  # normalize theta to [-pi, pi]
        return mu_hat, sigma_hat

    def pretty_print(self, mu):
        print(f"Robot: {mu[0][0]}, {mu[1][0]}, {math.degrees(mu[2][0])}")
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
        # time.sleep(1)

    def main(self, end_x, end_y, block, mode):
        path = self.a_star(Node(end_x, end_y), block, mode)
        print(path)
        for point in path:
            target_x, target_y = point
            print(f"Target Point: ({target_x}, {target_y})")
            while not (min(target_x - 2, target_x + 2) <= self.mu[0][0] <= max(target_x - 2, target_x + 2) and min(target_y - 2, target_y + 2) <= self.mu[1][0] <= max(target_y - 2, target_y + 2)):
            #while round(self.mu[0][0]) != target_x or round(self.mu[1][0]) != target_y:
                dx, dy = target_x - self.mu[0][0], target_y - self.mu[1][0]
                remain_distance = math.hypot(dx, dy)
                target_theta = math.atan2(dy, dx)

                v_linear = 15
                vx_world = v_linear * math.cos(target_theta)
                vy_world = v_linear * math.sin(target_theta)
                vx = vx_world * math.cos(self.mu[2][0]) + vy_world * math.sin(self.mu[2][0])
                vy = -vx_world * math.sin(self.mu[2][0]) + vy_world * math.cos(self.mu[2][0])
                m1, m2, m3, m4 = kinematic_model(vx=vx, vy=vy, omega=0)
                self.mpi_control.setFourMotors(m1, m2, m3, m4)
                dt = remain_distance / v_linear
                time.sleep(dt)
                self.mpi_control.carStop()
                self.step(vx, vy, 0, dt)


def test_a_star_algorithm():
    # Initialize the robot
    robot = Robot()
    block = Block()

    # Define a target point
    target_x, target_y = -198, 191

    # Run the A* algorithm
    print("Running A* algorithm...")
    traced_path = robot.main(target_x, target_y, block, "safe")

    print("Traced Path:", traced_path)


# Run the test case
if __name__ == "__main__":
    test_a_star_algorithm()



