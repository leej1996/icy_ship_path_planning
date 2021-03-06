import math
import queue
import time
from multiprocessing import connection, Event, Queue
from typing import Tuple

import dubins
import numpy as np
from skimage import transform

from cost_map import CostMap
from path_smoothing import path_smoothing
from primitives import Primitives
from priority_queue import CustomPriorityQueue
from ship import Ship
from swath import Swath, update_swath
from utils import heading_to_world_frame, snap_to_lattice, get_points_on_path


class AStar:

    def __init__(self, g_weight: float, h_weight: float, cmap: CostMap,
                 primitives: Primitives, ship: Ship, first_initial_heading: float):
        self.g_weight = g_weight
        self.h_weight = h_weight
        self.cmap = cmap
        self.chan_h, self.chan_w = np.shape(self.cmap.cost_map)
        self.primitives = primitives
        self.ship = ship
        self.first_initial_heading = first_initial_heading

    def search(self, start: tuple, goal: tuple, swath_dict: Swath, smooth_path: bool = True):
        generation = 0  # number of nodes expanded
        openSet = {start: generation}  # point_set of nodes considered for expansion
        closedSet = []
        cameFrom = {start: None}
        cameFrom_by_edge = {start: None}
        # cost from start
        g_score = {start: 0}
        # f score (g score + heuristic) (estimation of cost to goal)
        f_score = {start: self.heuristic(start, goal)}
        # path length between nodes
        path_length = {start: 0}
        # priority queue of all visited node f scores
        f_score_open_sorted = CustomPriorityQueue()
        f_score_open_sorted.put((start, f_score[start]))  # put item in priority queue

        while len(openSet) != 0:
            node = f_score_open_sorted.get()[0]

            if self.dist(node, goal) < 5 and abs(node[2] - goal[2]) < 0.01:
                # print("goal", goal)
                print("node", node)
                # print("Found path")

                # goal is not exactly the same as node, so when we search for goal (key)
                # in the dictionary, it has to be the same as node
                goal = node
                path = []
                new_path_length = []
                path.append(node)
                new_path_length.append(path_length[node])

                while node != start:
                    pred = cameFrom[node]
                    node = pred
                    path.append(node)
                    new_path_length.append(path_length[node])

                orig_path = path.copy()

                if smooth_path:
                    path.reverse()  # path: start -> goal
                    new_path_length.reverse()
                    # print("path", path)
                    add_nodes = int(len(path))  # number of nodes to add in the path smoothing algorithm

                    # cap at adding 10 nodes to reduce run time
                    add_nodes = min(add_nodes, 10)
                    # t0 = time.clock()
                    smooth_path, x1, y1, x2, y2 = path_smoothing(
                        path, new_path_length, self.cmap, start, goal, self.ship,
                        add_nodes, self.primitives.num_headings, dist_cuttoff=50
                    )
                    # t1 = time.clock() - t0
                    # print("smooth time", t1)
                else:
                    smooth_path = path
                    x1 = []
                    y1 = []
                    x2 = 0
                    y2 = 0
                    for vi in path:
                        x1.append(vi[0])
                        y1.append(vi[1])

                print("g_score at goal", g_score[goal])
                return True, smooth_path, closedSet, x1, y1, x2, y2, orig_path

            openSet.pop(node)
            closedSet.append(node)

            # find the base heading (e.g. cardinal or ordinal)
            num_base_h = self.primitives.num_headings // 4
            arr = np.asarray([
                (node[2] + num_base_h - h[2]) % num_base_h for h in self.primitives.edge_set_dict.keys()
            ])
            base_heading = np.argwhere(arr == 0)[0, 0]

            # get the edge set based on the current node heading
            edge_set = self.primitives.edge_set_dict[(0, 0, base_heading)]

            for e in edge_set:
                neighbour = self.concat(node, e, base_heading, self.primitives.num_headings)
                # print("NEIGHBOUR",neighbour)

                if neighbour[0] - self.ship.max_ship_length / 2 >= 0 and \
                        neighbour[0] + self.ship.max_ship_length / 2 <= self.chan_w and \
                        neighbour[1] - self.ship.max_ship_length / 2 >= 0 and \
                        neighbour[1] + self.ship.max_ship_length / 2 < self.chan_h:
                    # check if point is in closed point_set
                    neighbour_in_closed_set, closed_set_neighbour = self.is_point_in_set(neighbour, closedSet)
                    if neighbour_in_closed_set:
                        continue

                    # If near obstacle, check cost map to find cost of swath
                    if self.near_obstacle(node, self.cmap.cost_map.shape, self.cmap.obstacles,
                                          threshold=self.ship.max_ship_length * 3):
                        swath = self.get_swath(e, node, swath_dict)
                        if type(swath) == str and swath == "Fail":
                            continue
                        mask = self.cmap.cost_map[swath]
                        swath_cost = np.sum(mask)
                    else:
                        swath_cost = 0

                    temp_path_length = self.heuristic(node, neighbour)
                    cost = swath_cost + temp_path_length
                    temp_g_score = g_score[node] + cost
                    # print("cost", cost)

                    # check if point is in open set
                    neighbour_in_open_set, open_set_neighbour = self.is_point_in_set(neighbour, openSet)

                    if not neighbour_in_open_set:
                        heuristic_value = self.heuristic(neighbour, goal)
                        openSet[neighbour] = generation
                        cameFrom[neighbour] = node
                        cameFrom_by_edge[neighbour] = e
                        path_length[neighbour] = temp_path_length
                        # heading_delta[neighbour] = abs(neighbour[2] - node[2])
                        g_score[neighbour] = temp_g_score
                        f_score[neighbour] = self.g_weight * g_score[neighbour] + self.h_weight * heuristic_value
                        f_score_open_sorted.put((neighbour, f_score[neighbour]))
                    elif temp_g_score < g_score[open_set_neighbour]:
                        open_set_neighbour_heuristic_value = self.heuristic(open_set_neighbour, goal)
                        cameFrom[open_set_neighbour] = node
                        cameFrom_by_edge[open_set_neighbour] = e
                        path_length[open_set_neighbour] = temp_path_length
                        g_score[open_set_neighbour] = temp_g_score

                        new_f_score = self.g_weight * g_score[open_set_neighbour] + \
                                      self.h_weight * open_set_neighbour_heuristic_value
                        f_score_open_sorted._update((open_set_neighbour, f_score[open_set_neighbour]), new_f_score)
                        f_score[open_set_neighbour] = new_f_score
            generation += 1
        print("\nFail")
        return False, 'Fail', 'Fail', 'Fail', 'Fail', 'Fail', 'Fail', 'Fail'

    # helper methods
    def get_swath(self, e, start_pos, swath_dict: Swath):
        swath = np.zeros_like(self.cmap.cost_map, dtype=bool)
        heading = int(start_pos[2])
        # raw_swath = swath_set[tuple(e), heading]
        # print(((self.first_initial_heading - self.ship.initial_heading) * (180 / math.pi)))
        raw_swath = transform.rotate(swath_dict[tuple(e), heading],
                                     ((self.first_initial_heading - self.ship.initial_heading) * (180 / math.pi)))
        # swath mask has starting node at the centre and want to put at the starting node of currently expanded node
        # in the cmap, need to remove the extra columns/rows of the swath mask
        max_val = int(self.primitives.max_prim + self.ship.max_ship_length // 2)
        swath_size = raw_swath.shape[0]
        min_y = int(start_pos[1]) - max_val
        max_y = int(start_pos[1]) + max_val + 1
        min_x = int(start_pos[0]) - max_val
        max_x = int(start_pos[0]) + max_val + 1

        # Too far to the right
        invalid = False
        if max_x >= self.chan_w:
            overhang = max_x - (self.chan_w - 1)
            remove = raw_swath[:, slice(swath_size - overhang, swath_size)]
            if remove.sum() > 0:
                invalid = True
            raw_swath = np.delete(raw_swath, slice(swath_size - overhang, swath_size), axis=1)
            max_x = self.chan_w - 1
        # Too far to the left
        if min_x < 0:
            overhang = abs(min_x)
            remove = raw_swath[:, slice(0, overhang)]
            if remove.sum() > 0:
                invalid = True
            raw_swath = np.delete(raw_swath, slice(0, overhang), axis=1)
            min_x = 0
        # Too close to the top
        if max_y >= self.chan_h:
            overhang = max_y - (self.chan_h - 1)
            remove = raw_swath[slice(swath_size - overhang, swath_size), :]
            if remove.sum() > 0:
                invalid = True
            raw_swath = np.delete(raw_swath, slice(swath_size - overhang, swath_size), axis=0)
            max_y = self.chan_h - 1
        # Too close to the bottom
        if min_y < 0:
            overhang = abs(min_y)
            remove = raw_swath[slice(0, overhang), :]
            if remove.sum() > 0:
                invalid = True
            raw_swath = np.delete(raw_swath, slice(0, overhang), axis=0)
            min_y = 0
        swath[min_y:max_y, min_x:max_x] = raw_swath

        if invalid:
            return "Fail"
        else:
            return swath

    def heuristic(self, p_initial: Tuple, p_final: Tuple) -> float:
        """
        The Dubins' distance from initial to final points.
        """
        theta_0 = heading_to_world_frame(p_initial[2], self.ship.initial_heading, self.primitives.num_headings)
        theta_1 = heading_to_world_frame(p_final[2], self.ship.initial_heading, self.primitives.num_headings)
        p1 = (p_initial[0], p_initial[1], theta_0)
        p2 = (p_final[0], p_final[1], theta_1)
        path = dubins.shortest_path(p1, p2, self.ship.turning_radius)

        return path.path_length()

    @staticmethod
    def concat(x: Tuple, y: Tuple, base_heading: int, num_headings: int) -> Tuple:
        """
        given two points x,y in the lattice, find the concatenation x + y
        """
        # compute the spacing between base headings
        spacing = 2 * math.pi / num_headings

        # find the position and heading of the two points
        p1 = [x[0], x[1]]
        p1_theta = x[2] * spacing - spacing * base_heading  # starting heading
        p2 = [y[0], y[1]]
        p2_theta = y[2] * spacing  # edge heading

        R = np.array([[math.cos(p1_theta), -math.sin(p1_theta)],
                      [math.sin(p1_theta), math.cos(p1_theta)]])
        multiplication = np.matmul(R, np.transpose(np.asarray(p2)))

        result = np.asarray(p1) + multiplication

        # compute the final heading after concatenating x and y
        heading = p2_theta + x[2] * spacing - spacing * base_heading
        while heading >= 2 * math.pi:
            heading = heading - 2 * math.pi
        heading = heading / spacing
        # assert abs(heading - int(heading)) < 1e-4, "heading '{:4f}' should be an integer between 0-7".format(heading)

        return result[0], result[1], int(heading)

    @staticmethod
    def is_point_in_set(point, point_set, tol=1e-1):
        for curr_point in point_set:
            if AStar.dist(point, curr_point) < tol and point[2] == curr_point[2]:
                return True, curr_point
        return False, False

    @staticmethod
    def near_obstacle(node, map_dim, list_of_obstacles, threshold=10):
        for obs in list_of_obstacles:
            # check if ship is within radius + threshold squares of the center of obstacle, then do swath
            if AStar.dist(node, obs['centre']) < obs['radius'] + threshold or \
                    node[0] < threshold or node[0] > map_dim[0] - threshold:
                return True
        return False

    @staticmethod
    def dist(a, b):
        # Euclidean distance
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# method to call AStar in multiprocessing context
def gen_path(
        queue_state: Queue, pipe_path: connection.Pipe, shutdown_event: Event,
        ship: Ship, prim: Primitives, costmap: CostMap, swath_dict: Swath, a_star: AStar,
        goal_pos: Tuple, horizon: int = np.inf, smooth_path: bool = False
) -> None:
    planner_times = []  # list to keep track of time it takes for a* to find a path at each step
    while not shutdown_event.is_set():
        try:
            state_data = queue_state.get(block=True, timeout=1)  # blocking call
            print('\nReceived state data!')

            # update ship initial heading
            ship.body.angle = state_data['ship_body_angle']
            ship.initial_heading = -ship.body.angle + a_star.first_initial_heading

            # get new ship pos and computed snapped goal
            ship_pos = state_data['ship_pos']
            curr_goal = (goal_pos[0], min(goal_pos[1], (ship_pos[1] + horizon)), goal_pos[2])
            snapped_goal = snap_to_lattice(
                ship_pos, curr_goal, ship.initial_heading, ship.turning_radius,
                prim.num_headings, abs_init_heading=ship.initial_heading
            )

            # update costmap
            costmap.cost_map = state_data['costmap']
            costmap.obstacles = state_data['obstacles']

            # update primitives and update swath
            prim.rotate(-ship.body.angle, orig=True)
            new_swath_dict = update_swath(theta=-ship.body.angle, swath_dict=swath_dict)

            # get a rough idea of planner speed
            print('Generating next path...')
            t0 = time.time()

            # compute path to goal
            _, new_path, nodes_visited, x1, y1, x2, y2, _ = \
                a_star.search(ship_pos, snapped_goal, new_swath_dict, smooth_path=smooth_path)

            # save time to list and update average frequency
            dt = time.time() - t0
            planner_times.append(dt)
            print('Time elapsed: ', dt)
            print('Average planner frequency: {:.4f} Hz'.format(1 / (sum(planner_times) / len(planner_times))))

            if new_path != 'Fail' and len(new_path) > 1:
                # sample points along path
                full_path = get_points_on_path(
                    new_path, prim.num_headings, ship.initial_heading, ship.turning_radius, eps=1e-3
                )
                # send new path and node information to pipe
                print('Sending...')
                pipe_path.send({  # blocking call
                    'path': np.asarray(full_path),
                    'path_nodes': (x1, y1),
                    'smoothing_nodes': (x2, y2),
                    'nodes_expanded': nodes_visited
                })
                print('Sent path!')

        except queue.Empty:
            # nothing in queue so try again
            time.sleep(0.001)

        except ValueError as err:
            print("Queue closed: {}".format(err))
            break

    pipe_path.close()
