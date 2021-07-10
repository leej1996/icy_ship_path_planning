import math
import os
import pickle
import random
from typing import List, Tuple

import cv2
import dubins
import numpy as np
from pymunk import Poly
from skimage import draw
from matplotlib import patches
import matplotlib.pyplot as plt

from ship import Ship
from utils import heading_to_world_frame


class CostMap:
    def __init__(self, n, m, obstacle_penalty):
        self.n = n
        self.m = m
        self.cost_map = np.zeros((n, m))
        self.obstacles = []
        self.grouped_obstacles = []
        self.obstacle_penalty = obstacle_penalty

        # apply a cost to the boundaries of the channel
        self.boundary_cost()

    def boundary_cost(self, exp_factor=2.0, cutoff_factor=0.25) -> None:
        for col in range(self.m):
            self.cost_map[:, col] = max(0, (np.abs(col - self.m // 2) - cutoff_factor * self.m)) ** exp_factor

    def generate_obstacles(self, start_pos, goal_pos, num_obs, min_r, max_r,
                           upper_offset, lower_offset, allow_overlap=True, debug=False) -> List[dict]:
        channel_width = self.cost_map.shape[1]
        iteration_cap = 0
        while len(self.obstacles) < num_obs:
            near_obs = False
            x = random.randint(max_r, channel_width - max_r - 1)
            y = random.randint(start_pos[1] + lower_offset + max_r, int(goal_pos[1] - upper_offset - max_r))
            r = random.randint(min_r, max_r)

            if not allow_overlap:
                # check if obstacles overlap
                # NOTE: for this step we approximate the obstacles as circles
                for obs in self.obstacles:
                    if ((x - obs['centre'][0]) ** 2 + (y - obs['centre'][1]) ** 2) ** 0.5 < obs['radius'] + r:
                        near_obs = True
                        break

            if not near_obs:
                # generate polygon
                polygon = self.generate_polygon(diameter=r * 2, origin=(x, y))

                # compute radius from polygon (note this might be slightly different than the original sampled r)
                r = self.compute_polygon_diameter(polygon) / 2

                # compute the cost and update the costmap
                if self.populate_costmap(centre_coords=(x, y), radius=r, polygon=polygon):
                    # add the polygon to obstacles list if it is feasible
                    self.obstacles.append({
                        'vertices': polygon,
                        'centre': (x, y),
                        'radius': r
                    })

                # if debug mode then plot the polygon, its centre and the updated costmap
                if debug:
                    print('x', x, '\ny', y, '\nr', r, '\npolygon', polygon)

                    # polygon and centre
                    fig, ax = plt.subplots()
                    ax.plot(x, y, 'rx')
                    poly = patches.Polygon(polygon, True)
                    ax.add_patch(poly)
                    plt.show()

                    # costmap
                    plt.imshow(self.cost_map)
                    plt.show()

            iteration_cap += 1
            if iteration_cap > 300:
                break

        if allow_overlap:
            self.group_polygons()
        else:
            self.grouped_obstacles = self.obstacles

        return self.obstacles

    @staticmethod
    def generate_polygon(diameter, origin=(0, 0), num_vertices_range=(5, 10)) -> np.ndarray:
        """ algorithm described here https://cglab.ca/~sander/misc/ConvexGeneration/convex.html """
        # sample a random number for number of vertices
        num_vertices = random.randint(*num_vertices_range)

        # generate two lists of x and y of N random integers between 0 and n
        x = [random.uniform(0, diameter) for _ in range(num_vertices)]
        y = [random.uniform(0, diameter) for _ in range(num_vertices)]

        # sort both lists
        x.sort()
        y.sort()

        x_max = x[-1]
        y_max = y[-1]
        x_min = x[0]
        y_min = y[0]

        lastTop = x_min
        lastBot = x_min
        xVec = []

        for i in range(1, num_vertices - 1):
            val = x[i]
            if bool(random.getrandbits(1)):
                xVec.append(val - lastTop)
                lastTop = val
            else:
                xVec.append(lastBot - val)
                lastBot = val

        xVec.append(x_max - lastTop)
        xVec.append(lastBot - x_max)

        lastLeft = y_min
        lastRight = y_min
        yVec = []

        for i in range(1, num_vertices - 1):
            val = y[i]
            if bool(random.getrandbits(1)):
                yVec.append(val - lastLeft)
                lastLeft = val
            else:
                yVec.append(lastRight - val)
                lastRight = val

        yVec.append(y_max - lastLeft)
        yVec.append(lastRight - y_max)
        random.shuffle(yVec)

        pairs = zip(xVec, yVec)
        sorted_pairs = sorted(pairs, key=lambda pair: math.atan2(pair[0], pair[1]))

        minPolygonX = 0
        minPolygonY = 0
        x = 0
        y = 0
        points = []

        for pair in sorted_pairs:
            points.append((x, y))
            x += pair[0]
            y += pair[1]
            minPolygonX = min(minPolygonX, x)
            minPolygonY = min(minPolygonY, y)

        x_shift = x_min - minPolygonX
        y_shift = y_min - minPolygonY

        points = np.ceil(
            np.asarray(points) + np.array([x_shift, y_shift]).T
        )

        # find the centroid of polygon
        centre_pos = np.round(points.sum(axis=0) / len(points))

        # make centre of polygon at the origin
        points -= centre_pos - np.asarray(origin)  # assumes origin is in the 1st quadrant (positive x,y)

        # n x 2 array where each element is a vertex (x, y)
        return points

    def populate_costmap(self, centre_coords, radius, polygon, exp_factor=1.4) -> bool:
        row_coords = polygon[:, 1]
        col_coords = polygon[:, 0]

        centre_x = centre_coords[0]
        centre_y = centre_coords[1]

        # get all the cells occupied by polygon
        rr, cc = draw.polygon(row_coords, col_coords, shape=self.cost_map.shape)

        # if polygon shape is infeasible then ignore it
        if len(rr) == 0 or len(cc) == 0:
            return False

        for (row, col) in zip(rr, cc):
            dist = np.sqrt((row - centre_y) ** 2 + (col - centre_x) ** 2)
            new_cost = max(0, ((2 * radius - dist) ** exp_factor + 1) * self.obstacle_penalty)
            old_cost = self.cost_map[row, col]
            self.cost_map[row, col] = max(new_cost, old_cost)

        return True

    def group_polygons(self) -> None:
        dummy_costmap = np.zeros((self.n, self.m), dtype=np.uint8)

        for obs in self.obstacles:
            row_coords = obs["vertices"][:, 1]
            col_coords = obs["vertices"][:, 0]

            rr, cc = draw.polygon(row_coords, col_coords, shape=self.cost_map.shape)
            dummy_costmap[rr, cc] = 1

        contours, hierarchy = cv2.findContours(dummy_costmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            self.grouped_obstacles.append({
                "vertices": cont[:, 0]
            })

    def compute_path_cost(self, path: List, ship: Ship, num_headings: int, reverse_path=False, eps=1e0) -> Tuple[int, int]:
        if reverse_path:
            path.reverse()

        total_path_length = 0
        total_swath = np.zeros_like(self.cost_map, dtype=bool)
        for i, vi in enumerate(path[:-1]):
            vj = path[i + 1]
            # determine cost between node vi and vj  # FIXME: code duplication with generate_swath and path smoothing
            theta_0 = heading_to_world_frame(vi[2], ship.initial_heading, num_headings)
            theta_1 = heading_to_world_frame(vj[2], ship.initial_heading, num_headings)
            dubins_path = dubins.shortest_path((vi[0], vi[1], theta_0),
                                               (vj[0], vj[1], theta_1),
                                               ship.turning_radius - eps)

            configurations, _ = dubins_path.sample_many(1.2)

            # for each point sampled on dubins path, get x, y, theta
            for config in configurations:
                x_cell = int(round(config[0]))
                y_cell = int(round(config[1]))

                theta = config[2] - ship.initial_heading
                R = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

                # rotate/translate vertices of ship from origin to sampled point with heading = theta
                rot_vi = np.round(np.array([[x_cell], [y_cell]]) + R @ ship.vertices.T).astype(int)

                # draw rotated ship polygon and put occupied cells into a mask
                rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :], shape=self.cost_map.shape)
                total_swath[rr, cc] = True

            # update path length
            total_path_length += dubins_path.path_length()

        total_path_cost = self.cost_map[total_swath].sum() + total_path_length
        return total_path_cost, total_path_length

    def update(self, obstacles: List[Poly]) -> None:
        # clear costmap and obstacles
        self.cost_map[:] = 0
        self.obstacles = []
        # apply a cost to the boundaries of the channel
        self.boundary_cost()

        # update obstacles based on new positions
        for obs in obstacles:
            poly_vertices = np.asarray(
                [v.rotated(-obs.body.angle) + obs.body.position for v in obs.get_vertices()]
            ).astype(int)

            # recompute the obstacle radius
            r = self.compute_polygon_diameter(poly_vertices) / 2

            # compute the cost and update the costmap
            if self.populate_costmap(centre_coords=list(obs.body.position), radius=r, polygon=poly_vertices):
                # add the polygon to obstacles list if it is feasible
                self.obstacles.append({
                    'vertices': poly_vertices,
                    'centre': list(obs.body.position),
                    'radius': r
                })

    def save_to_disk(self) -> None:
        save_costmap_file = input("\n\nFile name to save out costmap (press enter to ignore)\n").lower()
        if save_costmap_file:
            fp = os.path.join("sample_costmaps", save_costmap_file + ".pk")
            with open(fp, "wb") as fd:
                pickle.dump(self, fd)
                print("Successfully saved costmap object to file path '{}'".format(fp))

    @staticmethod
    def compute_polygon_diameter(vertices) -> float:
        dist = lambda a, b: np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return max(dist(a, b) for a in vertices for b in vertices)


def main():
    # initialize costmap
    costmap = CostMap(n=50, m=50, obstacle_penalty=10)

    # params
    start_pos = (1, 1, 0)
    goal_pos = (20, 49, 0)
    num_obstacles = 1
    min_radius = 10
    max_radius = 20
    upper_offset = 1
    lower_offset = 1

    # generate random obstacles
    costmap.generate_obstacles(start_pos, goal_pos, num_obstacles,
                               min_radius, max_radius, upper_offset, lower_offset, debug=True)


if __name__ == "__main__":
    # run main to test costmap generation
    main()
