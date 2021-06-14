import time
import math

import dubins
import numpy as np

from cost_map import CostMap
from path_smoothing import path_smoothing
from primitives import Primitives
from priority_queue import CustomPriorityQueue


class AStar:

    def __init__(self, g_weight: float, h_weight: float, cmap: CostMap,
                 primitives: Primitives, ship_vertices: np.ndarray):
        self.g_weight = g_weight
        self.h_weight = h_weight
        self.cmap = cmap
        self.chan_h, self.chan_w = np.shape(self.cmap.cost_map)
        self.primitives = primitives
        self.ship_vertices = ship_vertices
        # compute ship length
        dist = lambda a, b: abs(a[0] - a[1])
        self.max_ship_length = np.ceil(max(dist(a, b) for a in ship_vertices for b in ship_vertices)).astype(int)
        assert self.max_ship_length != 0, 'ship length cannot be 0'

    def search(self, start, goal, turning_radius, cardinal_swath, ordinal_swath):
        # theta is measured ccw from y axis
        free_path_interval = 1
        generation = 0  # number of nodes expanded
        print("start", start)
        openSet = {start: generation}  # point_set of nodes considered for expansion
        closedSet = []
        cameFrom = {start: None}
        cameFrom_by_edge = {start: None}
        # cost from start
        g_score = {start: 0}
        # f score (g score + heuristic) (estimation of cost to goal)
        f_score = {start: self.heuristic(start, goal, turning_radius)}
        # path length between nodes
        path_length = {start: 0}
        # priority queue of all visited node f scores
        f_score_open_sorted = CustomPriorityQueue()
        f_score_open_sorted.put((start, f_score[start]))  # put item in priority queue

        while len(openSet) != 0:
            node = f_score_open_sorted.get()[0]

            # If ship past all obstacles, calc direct dubins path to goal
            if generation % free_path_interval == 0 and node != goal:
                past_all_obs = all(
                    self.past_obstacle(node, obs) for obs in self.cmap.obstacles
                )

                if past_all_obs:
                    print("Found path to goal")
                    cameFrom[goal] = node
                    path_length[goal] = self.heuristic(node, goal, turning_radius)
                    f_score[goal] = g_score[node] + path_length[goal]
                    pred = goal
                    node = pred

            if node == goal:
                print("Found path")
                path = []
                new_path_length = []
                print("goal", goal)
                cameFrom[goal] = cameFrom[node]
                path.append(goal)
                new_path_length.append(path_length[goal])

                while node != start:
                    pred = cameFrom[node]
                    node = pred
                    path.append(node)
                    new_path_length.append(path_length[node])

                path.reverse()  # path: start -> goal
                new_path_length.reverse()
                print("path", path)
                add_nodes = int(len(path))  # number of nodes to add in the path smoothing algorithm

                orig_path = path.copy()
                orig_cost = f_score[goal]
                t0 = time.clock()
                smooth_path, x1, y1, x2, y2 = path_smoothing(path, new_path_length, self.cmap.cost_map, turning_radius,
                                                             start, goal, add_nodes, self.ship_vertices,
                                                             dist_cuttoff=100)
                t1 = time.clock() - t0
                print("smooth time", t1)

                return True, orig_cost, smooth_path, closedSet, x1, y1, x2, y2, orig_path

            openSet.pop(node)
            closedSet.append(node)

            if (node[2] * 45) % 90 == 0:
                edge_set = self.primitives.edge_set_cardinal
                swath_set = cardinal_swath
            else:
                edge_set = self.primitives.edge_set_ordinal
                swath_set = ordinal_swath

            for e in edge_set:
                neighbour = self.concat(node, e)

                if 0 <= neighbour[0] < self.chan_w and 0 <= neighbour[1] < self.chan_h:
                    # check if point is in closed point_set
                    neighbour_in_closed_set, closed_set_neighbour = self.is_point_in_set(neighbour, closedSet)
                    if neighbour_in_closed_set:
                        continue

                    # If near obstacle, check cost map to find cost of swath
                    if self.near_obstacle(node, self.cmap.cost_map.shape, self.cmap.obstacles,
                                          threshold=self.max_ship_length * 3):
                        swath = self.get_swath(e, node, swath_set)
                        mask = self.cmap.cost_map[swath]
                        swath_cost = np.sum(mask)
                    else:
                        swath_cost = 0

                    temp_path_length = self.heuristic(node, neighbour, turning_radius)
                    cost = swath_cost + temp_path_length
                    temp_g_score = g_score[node] + cost

                    if neighbour in openSet:
                        neighbour_in_open_set = True
                        open_set_neighbour = neighbour
                    else:
                        neighbour_in_open_set = False
                        open_set_neighbour = False

                    if not neighbour_in_open_set:
                        heuristic_value = self.heuristic(neighbour, goal, turning_radius)
                        openSet[neighbour] = generation
                        cameFrom[neighbour] = node
                        cameFrom_by_edge[neighbour] = e
                        path_length[neighbour] = temp_path_length
                        # heading_delta[neighbour] = abs(neighbour[2] - node[2])
                        g_score[neighbour] = temp_g_score
                        f_score[neighbour] = self.g_weight * g_score[neighbour] + self.h_weight * heuristic_value
                        f_score_open_sorted.put((neighbour, f_score[neighbour]))
                    elif temp_g_score < g_score[open_set_neighbour]:
                        open_set_neighbour_heuristic_value = self.heuristic(open_set_neighbour, goal, turning_radius)
                        cameFrom[open_set_neighbour] = node
                        cameFrom_by_edge[open_set_neighbour] = e
                        path_length[open_set_neighbour] = temp_path_length
                        g_score[open_set_neighbour] = temp_g_score

                        new_f_score = self.g_weight * g_score[open_set_neighbour] + \
                                      self.h_weight * open_set_neighbour_heuristic_value
                        f_score_open_sorted._update((open_set_neighbour, f_score[open_set_neighbour]), new_f_score)
                        f_score[open_set_neighbour] = new_f_score
            generation += 1
        return False, 'Fail', 'Fail', 'Fail'

    def get_swath(self, e, start_pos, swath_set):
        swath = np.zeros_like(self.cmap.cost_map, dtype=bool)
        heading = int(start_pos[2])
        raw_swath = swath_set[tuple(e), heading]

        # swath mask has starting node at the centre and want to put at the starting node of currently expanded node
        # in the cmap, need to remove the extra columns/rows of the swath mask
        max_val = int(self.primitives.max_prim + self.max_ship_length)
        swath_size = raw_swath.shape[0]
        min_y = start_pos[1] - max_val
        max_y = start_pos[1] + max_val + 1
        min_x = start_pos[0] - max_val
        max_x = start_pos[0] + max_val + 1

        # Too far to the right
        if max_x >= self.chan_w:
            overhang = max_x - (self.chan_w - 1)
            raw_swath = np.delete(raw_swath, slice(swath_size - overhang, swath_size), axis=1)
            max_x = self.chan_w - 1
        # Too far to the left
        if min_x < 0:
            overhang = abs(min_x)
            raw_swath = np.delete(raw_swath, slice(0, overhang), axis=1)
            min_x = 0
        # Too close to the top
        if max_y >= self.chan_h:
            overhang = max_y - (self.chan_h - 1)
            raw_swath = np.delete(raw_swath, slice(swath_size - overhang, swath_size), axis=0)
            max_y = self.chan_h - 1
        # Too close to the bottom
        if min_y < 0:
            overhang = abs(min_y)
            raw_swath = np.delete(raw_swath, slice(0, overhang), axis=0)
            min_y = 0
        swath[min_y:max_y, min_x:max_x] = raw_swath

        return swath

    # helper methods
    @staticmethod
    def heuristic(p_initial, p_final, turning_radius):
        """
        The Dubins' distance from initial to final points.
        """
        p1 = (p_initial[0], p_initial[1], math.radians((p_initial[2] * 45 + 90) % 360))
        p2 = (p_final[0], p_final[1], math.radians((p_final[2] * 45 + 90) % 360))
        path = dubins.shortest_path(p1, p2, turning_radius)
        return path.path_length()

    @staticmethod
    def concat(x, y):
        """
        given two points x,y in the lattice, find the concatenation x + y
        """
        rot = x[2] * math.pi / 4  # starting heading
        p1 = [x[0], x[1]]
        p2_theta = y[2] * math.pi / 4  # edge heading
        p2 = [y[0], y[1]]

        # cardinal
        heading = p2_theta + rot
        if x[2] % 2 != 0:
            # ordinal
            rot = rot - math.pi / 4

        R = np.array([[math.cos(rot), -math.sin(rot)],
                      [math.sin(rot), math.cos(rot)]])
        multiplication = np.matmul(R, np.transpose(np.asarray(p2)))

        result = np.asarray(p1) + multiplication

        while heading >= 2 * math.pi:
            heading = heading - 2 * math.pi
        heading = heading / (math.pi / 4)

        return int(result[0]), int(result[1]), int(round(heading))

    @staticmethod
    def is_point_in_set(point, point_set):
        for curr_point in point_set:
            if AStar.dist(point, curr_point) < 0.001 and abs(point[2] - curr_point[2]) < 0.001:
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
    def past_obstacle(node, obs):
        # also check if ship is past all obstacles (under assumption that goal is always positive y direction from start)
        # obstacle y coord + radius
        return node[1] > obs['centre'][1] + obs['radius']  # doesn't account for channel boundaries

    @staticmethod
    def dist(a, b):
        # Euclidean distance
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5