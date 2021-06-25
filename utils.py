import math

import dubins
import numpy as np


def heading_to_world_frame(heading: int, theta_0: float):
    """
    :param heading: ordinal or cardinal heading from ships frame of reference
    :param theta_0: angle between ship and fixed coordinates
    """
    return (heading * math.pi / 4 + theta_0) % (2 * math.pi)


def plot_path(ax, path, cost_map, ship):
    x = []
    y = []

    for vi in path:
        x.append(vi[0])
        y.append(vi[1])
    ax.imshow(cost_map, origin='lower')
    for i in range(np.shape(path)[0] - 1):
        p1 = path[i]
        p2 = path[i + 1]
        theta_0 = heading_to_world_frame(p1[2], ship.initial_heading)
        theta_1 = heading_to_world_frame(p2[2], ship.initial_heading)
        dubins_path = dubins.shortest_path((p1[0], p1[1], theta_0),
                                           (p2[0], p2[1], theta_1),
                                           ship.turning_radius)
        configurations, _ = dubins_path.sample_many(0.2)
        x1 = []
        y1 = []
        for config in configurations:
            x1.append(config[0])
            y1.append(config[1])
        ax.plot(x1, y1, 'g')

    ax.plot(x, y, 'bx')

