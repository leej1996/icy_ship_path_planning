import math
from typing import Tuple, List

import dubins
import numpy as np


def heading_to_world_frame(heading: int, theta_0: float, num_headings: int):
    """
    :param heading: ordinal or cardinal heading from ships frame of reference
    :param theta_0: angle between ship and fixed coordinates
    """
    return (heading * 2 * math.pi / num_headings + theta_0) % (2 * math.pi)


def plot_path(ax, path, cost_map, ship, num_headings, eps=1e0):
    ax.imshow(cost_map, origin='lower')
    for i in range(np.shape(path)[0] - 1):
        p1 = path[i]
        p2 = path[i + 1]
        x, y, _ = get_points_on_dubins_path(p1, p2, num_headings, ship.initial_heading, ship.turning_radius, eps)
        ax.plot(x, y, 'g')

    x = [vi[0] for vi in path]
    y = [vi[1] for vi in path]
    ax.plot(x, y, 'bx')


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
