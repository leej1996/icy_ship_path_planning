from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import dubins
from utils import heading_to_world_frame


class Primitives:
    def __init__(self, scale: float = None, initial_heading: float = None):
        self.scale = scale
        self.initial_heading = initial_heading
        self.edge_set_cardinal = np.asarray([
            (1, 0, 0),
            (2, 1, 0),
            (2, -1, 0),
            (2, 1, 1),
            (2, -1, 7),
            (2, 2, 1),
            (2, -2, 7),
            (3, 0, 1),
            (3, 0, 7),
            (3, 2, 0),
            (3, -2, 0),
            (3, 3, 2),
            (3, -3, 6),
            (3, 4, 2),
            (3, -4, 6),
            (3, 5, 2),
            (3, -5, 6),
            (4, 5, 0),
            (4, -5, 0)
        ])
        self.edge_set_ordinal = np.asarray([
            (0, 3, 2),
            (0, 4, 3),
            (0, 5, 1),
            (0, 5, 3),
            (1, 1, 1),
            (1, 2, 1),
            (1, 2, 2),
            (1, 3, 1),
            (1, 4, 1),
            (2, 1, 0),
            (2, 1, 1),
            (2, 2, 0),
            (2, 2, 2),
            (3, 0, 0),
            (3, 1, 1),
            (4, 0, 7),
            (4, 1, 1),
            (5, 0, 1),
            (5, 0, 7)
        ])

        if self.scale is not None:
            self.edge_set_ordinal[:, :2] *= scale
            self.edge_set_cardinal[:, :2] *= scale

        self.rotate(theta=initial_heading)
        self.max_prim = self.get_max_prim()
        self.orig_edge_set_ordinal = self.edge_set_ordinal
        self.orig_edge_set_cardinal = self.edge_set_cardinal

    def view(self, theta: float = None, save_fig_prefix="primitives_", turning_radius: float = 1):
        """ plots all the primitives in the edge point_set """
        if theta is None:
            theta = self.initial_heading
        arrow_length = 0.2 * turning_radius
        for edge_set, name in zip([self.edge_set_ordinal, self.edge_set_cardinal],
                                  ['ordinal', 'cardinal']):
            # use an arrow to indicate node location and heading
            if name == 'ordinal':
                origin = (0, 0, 1)
            else:
                origin = (0, 0, 0)
            fig = plt.figure(figsize=(6, 6))
            plt.title("Theta = {} {}".format(round(theta, 5), name))
            arrow = partial(plt.arrow, head_width=0.2* turning_radius, width=0.05* turning_radius, ec="green")

            R = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            for item in edge_set:
                # compute the heading
                heading = item[2] * np.pi / 4
                xy = item[:2]
                dxdy = (R @ np.asarray([np.cos(heading), np.sin(heading)])) * arrow_length
                arrow(x=xy[0], y=xy[1], dx=dxdy[0], dy=dxdy[1])

                theta_0 = heading_to_world_frame(origin[2], theta) % (2 * np.pi)
                theta_1 = heading_to_world_frame(item[2], theta) % (2 * np.pi)
                dubins_path = dubins.shortest_path((origin[0], origin[1], theta_0),
                                                   (item[0], item[1], theta_1),
                                                   turning_radius - 1e-4)
                configurations, _ = dubins_path.sample_many(0.2)
                # 0.01
                x = []
                y = []

                for config in configurations:
                    x.append(config[0])
                    y.append(config[1])

                plt.plot(x, y, 'b')

            plt.savefig(save_fig_prefix + name + ".png")
            plt.show()

    def rotate(self, theta: float, orig: bool = False):
        R = np.asarray([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        if orig:
            self.edge_set_cardinal = self.orig_edge_set_cardinal @ R.T
            self.edge_set_ordinal = self.orig_edge_set_ordinal @ R.T
        else:
            self.edge_set_cardinal = self.edge_set_cardinal @ R.T
            self.edge_set_ordinal = self.edge_set_ordinal @ R.T

    def get_max_prim(self):
        # compute the total space occupied by the primitives
        prims = np.concatenate((self.edge_set_ordinal, self.edge_set_cardinal))
        max_x, min_x = prims[:, 0].max(), prims[:, 0].min()
        max_y, min_y = prims[:, 1].max(), prims[:, 1].min()
        return int(
            round(max(max_x, max_y, abs(min_x), abs(min_y)))
        )


if __name__ == '__main__':
    # for testing purposes
    initial_heading = 0
    p = Primitives(initial_heading=initial_heading)
    print(p.edge_set_cardinal)
    print(p.edge_set_ordinal)
    p.view()
