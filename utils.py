import math
from typing import Tuple, List

import dubins
import numpy as np
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
        x, y, theta = get_points_on_dubins_path(p1, p2, num_headings, ship.initial_heading, ship.turning_radius, eps)
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
                              turning_radius: float, eps: float = 0., step_size: float = 0.2) -> Tuple[List, List, List]:
    theta_0 = heading_to_world_frame(p1[2], initial_heading, num_headings)
    theta_1 = heading_to_world_frame(p2[2], initial_heading, num_headings)
    dubins_path = dubins.shortest_path((p1[0], p1[1], theta_0),
                                       (p2[0], p2[1], theta_1),
                                       turning_radius - eps)
    configurations, _ = dubins_path.sample_many(step_size)
    x = [item[0] for item in configurations]
    y = [item[1] for item in configurations]
    theta = [item[2] - initial_heading for item in configurations]

    return x, y, theta
