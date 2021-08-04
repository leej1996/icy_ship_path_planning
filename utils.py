import math
from typing import Tuple, List

import dubins
import numpy as np
import pymunk
from matplotlib.axes import Axes

from ship import Ship


def heading_to_world_frame(heading: int, theta_0: float, num_headings: int):
    """
    :param heading: ordinal or cardinal heading from ships frame of reference
    :param theta_0: angle between ship and fixed coordinates
    """
    return (heading * 2 * math.pi / num_headings + theta_0) % (2 * math.pi)


def plot_path(ax: Axes, path: List, cost_map: np.ndarray, ship: Ship,
              num_headings: int, return_points=False, eps: float = 1e0):
    # lists to keep track of the points sampled along the path
    p_x = []
    p_y = []
    p_theta = []
    for i in range(np.shape(path)[0] - 1):
        p1 = path[i]
        p2 = path[i + 1]
        x, y, theta, _ = get_points_on_dubins_path(p1, p2, num_headings, ship.initial_heading, ship.turning_radius, eps)
        p_x.extend(x)
        p_y.extend(y)
        p_theta.extend(theta)

    # lists to store the x and y positions of the nodes
    n_x = [vi[0] for vi in path]
    n_y = [vi[1] for vi in path]

    if return_points:
        return n_x, n_y, p_x, p_y, p_theta

    return [
        ax.imshow(cost_map, origin='lower'),
        *ax.plot(n_x, n_y, 'bx'),
        *ax.plot(p_x, p_y, 'g')
    ]


def get_points_on_dubins_path(p1: Tuple, p2: Tuple, num_headings: int, initial_heading: float,
                              turning_radius: float, eps: float = 0., step_size: float = 0.2) -> Tuple[List, List, List, float]:
    theta_0 = heading_to_world_frame(p1[2], initial_heading, num_headings)
    theta_1 = heading_to_world_frame(p2[2], initial_heading, num_headings)
    dubins_path = dubins.shortest_path((p1[0], p1[1], theta_0),
                                       (p2[0], p2[1], theta_1),
                                       turning_radius - eps)
    configurations, _ = dubins_path.sample_many(step_size)
    x = [item[0] for item in configurations]
    y = [item[1] for item in configurations]
    theta = [item[2] - initial_heading for item in configurations]

    return x, y, theta, dubins_path.path_length()


def get_points_on_path(path: List, num_headings: int, initial_heading: float, turning_radius: float,
                       show_prims: bool = False, eps: float = 1e-3) -> Tuple[List, List, List]:
    p_x, p_y, p_theta = [], [], []
    # reverse the path
    path = path[::-1]
    for i in range(np.shape(path)[0] - 1):
        p1 = path[i]
        p2 = path[i + 1]
        x, y, theta, _ = get_points_on_dubins_path(
            p1, p2, num_headings, initial_heading, turning_radius, eps
        )
        p_x.extend(x)
        p_y.extend(y)
        p_theta.extend(theta)

    return p_x, p_y, p_theta


def create_polygon(space, staticBody, vertices, x, y, density):
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Poly(body, vertices)
    shape.density = density
    space.add(body, shape)

    # create pivot constraint to simulate linear friction
    pivot = pymunk.constraints.PivotJoint(staticBody, body, (0, 0))
    pivot.max_bias = 0
    pivot.max_force = 10000.0

    # create gear constraint to simulate angular friction
    gear = pymunk.constraints.GearJoint(staticBody, body, 0, 1)
    gear.max_bias = 0
    gear.max_force = 5000.0
    space.add(pivot, gear)
    return shape


def snap_to_lattice(start_pos, goal_pos, initial_heading, turning_radius, num_headings,
                    abs_init_heading=None, abs_goal_heading=None):
    # compute the spacing between base headings
    spacing = 2 * math.pi / num_headings

    # Rotate goal to lattice coordinate system
    R = np.asarray([
        [np.cos(initial_heading), -np.sin(initial_heading)],
        [np.sin(initial_heading), np.cos(initial_heading)]
    ])

    # determine how far from lattice the goal position is
    difference = R @ np.array([[goal_pos[0] - start_pos[0]], [goal_pos[1] - start_pos[1]]])
    diff_y = difference[1][0] % turning_radius
    diff_x = difference[0][0] % turning_radius

    # determine difference in heading
    abs_init_heading = heading_to_world_frame(start_pos[2], initial_heading, num_headings) \
        if abs_init_heading is None else abs_init_heading
    abs_goal_heading = heading_to_world_frame(goal_pos[2], initial_heading, num_headings) \
        if abs_goal_heading is None else abs_goal_heading
    diff = abs_goal_heading - abs_init_heading

    if diff < 0:
        diff = diff + (2 * math.pi)

    # check if x,y coordinates or heading are off lattice
    if diff_y != 0 or diff_x != 0 or diff % spacing != 0:
        if diff_y >= turning_radius / 2:
            new_goal_y = difference[1][0] + turning_radius - diff_y
        elif diff_y == 0:
            new_goal_y = difference[1][0]  # no change
        else:
            new_goal_y = difference[1][0] - diff_y

        if diff_x >= turning_radius / 2:
            new_goal_x = difference[0][0] + turning_radius - diff_x
        elif diff_x == 0:
            new_goal_x = difference[0][0]
        else:
            new_goal_x = difference[0][0] - diff_x

        # round to nearest base heading
        new_theta = round(diff / spacing)
        if new_theta > num_headings - 1:
            new_theta -= num_headings

        # rotate coordinates back to original frame
        new_goal = np.array([[new_goal_x], [new_goal_y]])
        new_goal = R.T @ new_goal
        goal_pos = (
            new_goal[0][0] + start_pos[0],
            new_goal[1][0] + start_pos[1],
            new_theta
        )

    return goal_pos


class Path:
    """
    There are two path objects, the output from a star that the cost can be calculated from (planned_path),
    and the path with many more nodes that the ship actually follows (path).
    """

    def __init__(self, path: np.ndarray):
        self.path = path  # shape is 3 x n
        self.old_path_cnt = 0  # counters to keep track of how frequently old/new path is better
        self.new_path_cnt = 0

    def clip_path(self, ship_pos_y: float):
        # remove points along path that are less than ship y position
        return self.path[..., self.path[1] > ship_pos_y]


def rotation_matrix(theta) -> np.ndarray:
    return np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
