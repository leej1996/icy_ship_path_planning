import math
import random

import numpy as np
from skimage import draw
from matplotlib import patches
import matplotlib.pyplot as plt


class CostMap:
    def __init__(self, n, m, obstacle_penalty):
        self.cost_map = np.zeros((n, m))
        self.obstacles = []
        self.obstacle_penalty = obstacle_penalty

    def generate_obstacles(self, start_pos, goal_pos, num_obs, min_r, max_r, upper_offset, lower_offset, debug=False):
        channel_width = self.cost_map.shape[1]
        iteration_cap = 0
        while len(self.obstacles) < num_obs:
            near_obs = False
            x = random.randint(max_r, channel_width - max_r - 1)
            y = random.randint(start_pos[1] + lower_offset + max_r, goal_pos[1] - upper_offset - max_r)
            r = random.randint(min_r, max_r)

            # check if obstacles overlap, for this step we approximate the obstacles as circles
            for obs in self.obstacles:
                if ((x - obs['centre'][0]) ** 2 + (y - obs['centre'][1]) ** 2) ** 0.5 < obs['radius'] + r:
                    near_obs = True
                    break

            if not near_obs:
                # generate polygon
                polygon = self.generate_polygon(diameter=r*2, origin=(x, y))

                # compute the cost and update the costmap
                if self.populate_costmap(centre_coords=(x, y), polygon=polygon):
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

    def populate_costmap(self, centre_coords, polygon, exp_factor=1.3) -> bool:
        row_coords = polygon[:, 1]
        col_coords = polygon[:, 0]

        centre_x = centre_coords[0]
        centre_y = centre_coords[1]

        # get all the cells occupied by polygon
        rr, cc = draw.polygon(row_coords, col_coords)

        # if polygon shape is infeasible then ignore it
        if len(rr) == 0 or len(cc) == 0:
            return False

        # get all the cells occupied by polygon perimeter
        rr_perim, cc_perim = draw.polygon_perimeter(row_coords, col_coords)

        # set the cost for the cells along the perimeter
        self.cost_map[rr_perim, cc_perim] = self.obstacle_penalty

        # keep track of the cells which have their costs already computed
        cells_computed = {(row, col): [self.obstacle_penalty] for row, col in zip(rr_perim, cc_perim)}

        # for each edge in the polygon perimeter find the line that goes from the edge cell to the centre
        # and compute the costs of the cells along that line
        for (row_perim, col_perim) in zip(rr_perim, cc_perim):
            rr_line, cc_line = draw.line(row_perim, col_perim, centre_y, centre_x)
            dist_poly_edge_to_centre = np.sqrt((row_perim - centre_y) ** 2 + (col_perim - centre_x) ** 2)

            for (row, col) in zip(rr_line, cc_line):
                # skip the cell part of the polygon perimeter
                if (row, col) == (row_perim, col_perim):
                    continue

                if (row, col) not in cells_computed:
                    cells_computed[(row, col)] = []  # initialize new key-val pair to store cost

                # compute dist for current cell to centre
                dist_cell_to_centre = np.sqrt((centre_x - col) ** 2 + (centre_y - row) ** 2)
                cost = ((dist_poly_edge_to_centre - dist_cell_to_centre) ** exp_factor + 1) * self.obstacle_penalty
                cells_computed[(row, col)].append(cost)

        # update costmap
        for cell in cells_computed:
            self.cost_map[cell] = min(cells_computed[cell])

        # fill cells that are missing costs, find cost of 8 nearest neighbours and set cost to min
        for (row, col) in zip(rr, cc):
            if (row, col) not in cells_computed:
                nearest_nbrs = [
                    (row + i, col + j)
                    for i in [-1, 0, 1]
                    for j in [-1, 0, 1]
                    if i != 0 or j != 0
                ]

                # get the cost of the neighbours, excluding cost of 0
                cost_nbrs = [self.cost_map[cell] for cell in nearest_nbrs if self.cost_map[cell] != 0]
                # set the cost for missing as the mean of neighbours
                self.cost_map[row, col] = np.mean(cost_nbrs)

        # TODO: do some smoothing
        return True


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
