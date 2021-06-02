from functools import partial

import numpy as np
import matplotlib.pyplot as plt


class Primitives:
    def __init__(self, scale=None, rotate=True):
        self.scale = scale
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

        if rotate:
            self.rotate()

    @staticmethod
    def view(edge_set, save_fig_fp="primitives.png"):
        """ plots all the primitives in the edge set """
        # use an arrow to indicate node location and heading
        fig = plt.figure(figsize=(8, 4))
        arrow = partial(plt.arrow, head_width=0.2, width=0.05, ec="green")

        theta = np.pi/2
        arrow_length = 0.2
        R = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        for item in edge_set:
            # compute the heading
            heading = item[2] * np.pi/4
            xy = item[:2]
            # we consider positive x to be in right direction and positive y to be in the up direction
            dxdy = (R @ np.asarray([np.cos(heading), np.sin(heading)])) * arrow_length
            arrow(x=xy[0], y=xy[1], dx=dxdy[0], dy=dxdy[1])

        plt.savefig(save_fig_fp)
        plt.show()

    def rotate(self, theta=np.pi/2):
        R = np.asarray([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        self.edge_set_cardinal = np.round((self.edge_set_cardinal @ R.T)).astype(int)
        self.edge_set_ordinal = np.round((self.edge_set_ordinal @ R.T)).astype(int)
