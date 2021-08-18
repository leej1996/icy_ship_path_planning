import math
import os
import pickle
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pymunk import Poly
from skimage import draw


class CostMap:
    def __init__(self, n: int, m: int, obstacle_penalty: float, min_r: int = 1, max_r: int = 8,
                 inf_stream: bool = False, viewable_height: int = None, total_obs: int = None, new_obs_dist: int = None):
        self.n = n
        self.m = m
        self.cost_map = np.zeros((self.n, self.m))
        self.obstacles = []
        self.grouped_obstacles = []
        self.obstacle_penalty = obstacle_penalty

        # apply a cost to the boundaries of the channel
        self.boundary_cost()

        # min and max size of obstacles
        self.min_r = min_r
        self.max_r = max_r

        # these attributes are for inf stream
        self.inf_stream = inf_stream
        self.viewable_height = viewable_height  # the height of the costmap that is viewable
        self.total_obs = total_obs  # total number of obstacles
        self.new_obs_dist = new_obs_dist  # distance traveled before new obstacles are added
        self.new_obs_count = 0

    def boundary_cost(self, exp_factor=1.2, cutoff_factor=0.30) -> None:
        for col in range(self.m):
            self.cost_map[:, col] = max(
                0, (np.abs(col - self.m / 2) - cutoff_factor * self.m)
            ) ** exp_factor * self.obstacle_penalty

    def generate_obstacles(self, min_y, max_y, num_obs, upper_offset=0,
                           lower_offset=0, allow_overlap=False, debug=False) -> List[dict]:
        iteration_cap = 0
        obstacles = []
        while len(obstacles) < num_obs:
            near_obs = False
            x = random.randint(self.max_r, self.m - self.max_r - 1)
            y = random.randint(int(min_y + lower_offset), int(max_y - upper_offset))
            r = random.randint(self.min_r, self.max_r)

            if not allow_overlap:
                # check if obstacles overlap
                # NOTE: for this step we approximate the obstacles as circles
                for obs in obstacles:
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
                    obstacles.append({
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

        self.obstacles.extend(obstacles)  # add new obstacles to class level list

        if allow_overlap:
            self.group_polygons()
        else:
            self.grouped_obstacles = self.obstacles

        return obstacles  # return the newly added obstacles

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
        dummy_costmap = np.zeros_like(self.cost_map, dtype=np.uint8)

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

    def update(self, obstacles: List[Poly], ship_pos_y: float = None) -> Tuple[List[dict], List[Poly]]:
        # clear costmap and obstacles
        self.cost_map[:] = 0
        self.obstacles = []
        # apply a cost to the boundaries of the channel
        self.boundary_cost()

        to_remove = []  # list to keep of polys that need to be deleted
        # update obstacles based on new positions
        for obs in obstacles:
            poly_vertices = np.asarray(
                [v.rotated(-obs.body.angle) + obs.body.position for v in obs.get_vertices()]
            ).astype(int)

            # recompute the obstacle radius
            r = self.compute_polygon_diameter(poly_vertices) / 2

            # remove obstacle if it is out of sight behind ship
            if (
                self.inf_stream
                and self.new_obs_count > 2
                and obs.body.position.y
                < (ship_pos_y - self.new_obs_dist - r)
            ):
                to_remove.append(obs)

            # compute the cost and update the costmap if obstacle is feasible
            elif self.populate_costmap(centre_coords=list(obs.body.position), radius=r, polygon=poly_vertices):
                # add the polygon to obstacles list if it is feasible
                self.obstacles.append({
                    'vertices': poly_vertices,
                    'centre': list(obs.body.position),
                    'radius': r
                })
            else:
                # if poly is not on costmap then remove
                to_remove.append(obs)

        # check if need to delete any more obs and generate new ones if necessary
        to_add = []
        if self.inf_stream and ship_pos_y > self.new_obs_count * self.new_obs_dist:
            # compute number of new obs to add
            new_obs_num = self.total_obs - len(self.obstacles)

            if new_obs_num:
                start_pos_y = self.new_obs_count * self.new_obs_dist + self.viewable_height
                to_add = self.generate_obstacles(
                    min_y=start_pos_y, max_y=start_pos_y + self.new_obs_dist, num_obs=new_obs_num,
                )  # this does not guarantee returning the number of requested obstacles

            # update count
            self.new_obs_count += 1

        return to_add, to_remove

    def update2(self, obstacle_penalty: float):
        # update attribute
        self.obstacle_penalty = obstacle_penalty

        # clear costmap and obstacles
        self.cost_map[:] = 0
        # recompute cost to the boundaries of the channel
        self.boundary_cost()

        # re-populate costmap with new costs of obstacles
        for obs in self.obstacles:
            self.populate_costmap(centre_coords=list(obs['centre']), radius=obs['radius'], polygon=obs['vertices'])

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
    costmap = CostMap(n=320, m=50, obstacle_penalty=10, min_r=10, max_r=20)

    # params
    start_pos = (1, 1, 0)
    goal_pos = (20, 49, 0)
    num_obstacles = 1
    upper_offset = 1
    lower_offset = 1

    # generate random obstacles
    costmap.generate_obstacles(start_pos[1], goal_pos[1], num_obstacles,
                               upper_offset, lower_offset, debug=True)


if __name__ == "__main__":
    # run main to test costmap generation
    main()
