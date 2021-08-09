from functools import partial
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from utils import get_points_on_dubins_path, rotation_matrix


class Primitives:
    def __init__(self, scale: float = 1, initial_heading: float = 0, num_headings: int = 8 or 16):
        self.scale = scale
        self.num_headings = num_headings

        edge_set_dict = self.get_primitives(num_headings)
        # scale the edge sets and turn them into npy arrays
        self.edge_set_dict = {}
        for k, v in edge_set_dict.items():
            edge_set = np.asarray(v)
            edge_set[:, :2] *= scale
            self.edge_set_dict[k] = edge_set

        self.rotate(theta=initial_heading)
        self.max_prim = self.get_max_prim()
        self.orig_edge_set = self.edge_set_dict.copy()

    def view(self, theta: float = 0, save_fig_prefix="primitives_", turning_radius: float = None, eps=1e-10):
        """ plots all the primitives in the edge set dict """
        if turning_radius is None:
            turning_radius = self.scale
        arrow_length = 0.2 * turning_radius
        for origin, edge_set in self.edge_set_dict.items():
            # use an arrow to indicate node location and heading
            fig = plt.figure(figsize=(6, 6))
            plt.title("Theta: {} (rad)\nStart heading: {}".format(round(theta, 5), origin[2]))
            arrow = partial(plt.arrow, head_width=0.2 * turning_radius, width=0.05 * turning_radius, ec="green")

            R = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            for edge in edge_set:
                # compute the heading
                heading = edge[2] * 2 * np.pi / self.num_headings
                xy = edge[:2]
                dxdy = (R @ np.asarray([np.cos(heading), np.sin(heading)])) * arrow_length
                arrow(x=xy[0], y=xy[1], dx=dxdy[0], dy=dxdy[1])

                x, y, _, _ = get_points_on_dubins_path(origin, edge, self.num_headings, theta, turning_radius, eps)
                plt.plot(x, y, 'b')

            # plt.savefig(save_fig_prefix + str(origin[2]) + ".png")
            plt.show()

    def rotate(self, theta: float, orig: bool = False):
        R = rotation_matrix(theta)

        edge_set = self.orig_edge_set if orig else self.edge_set_dict
        self.edge_set_dict = {k: v @ R.T for k, v in edge_set.items()}

    def get_max_prim(self):
        prims = np.concatenate(list(self.edge_set_dict.values()))
        max_x, min_x = prims[:, 0].max(), prims[:, 0].min()
        max_y, min_y = prims[:, 1].max(), prims[:, 1].min()
        return int(
            round(max(max_x, max_y, abs(min_x), abs(min_y)))
        )

    @staticmethod
    def get_primitives(num_headings) -> Dict[Tuple, List]:
        if num_headings == 8:
            return {
                (0, 0, 0): [
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
                ],
                (0, 0, 1): [
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
                ]
            }
        elif num_headings == 16:
            return {
                (0, 0, 0): [
                    (1, 0, 0),
                    (1, 0, 1),
                    (1, 0, 15),
                    (1, 1, 4),
                    (1, -1, 12),
                    (2, 0, 2),
                    (2, 0, 14),
                    (2, 1, 0),
                    (2, -1, 0),
                    (2, 1, 1),
                    (2, -1, 15),
                    (2, 2, 1),
                    (2, 2, 2),
                    (2, 2, 3),
                    (2, 2, 4),
                    (2, -2, 12),
                    (2, -2, 13),
                    (2, -2, 14),
                    (2, -2, 15),
                    (2, -3, 0),
                    (2, 3, 1),
                    (2, 3, 2),
                    (2, -3, 14),
                    (2, -3, 15),
                    (2, -4, 0),
                    (3, 0, 3),
                    (3, 0, 13),
                    (3, -1, 2),
                    (3, 1, 14),
                    (3, -3, 1),
                    (3, 3, 15),
                    (4, -1, 3),
                    (4, 1, 13)
                ],
                (0, 0, 1): [
                    (-1, 4, 4),
                    (0, 3, 3),
                    (0, 3, 4),
                    (0, 3, 5),
                    (0, 3, 6),
                    (0, 4, 2),
                    (1, 0, 0),
                    (1, 0, 15),
                    (1, 1, 2),
                    (1, 1, 3),
                    (1, 2, 2),
                    (1, 2, 3),
                    (1, 3, 1),
                    (1, 3, 2),
                    (2, 0, 14),
                    (2, 1, 0),
                    (2, 1, 1),
                    (2, 1, 2),
                    (2, 1, 3),
                    (2, 1, 15),
                    (2, -1, 15),
                    (2, 2, 0),
                    (2, 2, 4),
                    (2, -2, 12),
                    (2, 3, 0),
                    (3, 0, 3),
                    (3, 0, 13),
                    (3, -2, 1)
                ],
                (0, 0, 2): [
                    (-1, 3, 4),
                    (-1, 4, 2),
                    (0, 2, 4),
                    (0, 2, 5),
                    (0, 2, 6),
                    (0, 3, 2),
                    (0, 3, 6),
                    (0, 3, 7),
                    (0, 4, 1),
                    (1, 1, 1),
                    (1, 1, 2),
                    (1, 1, 3),
                    (1, 2, 1),
                    (1, 2, 2),
                    (1, 2, 3),
                    (1, 3, 1),
                    (1, 3, 2),
                    (2, 0, 0),
                    (2, 0, 14),
                    (2, 0, 15),
                    (2, 1, 1),
                    (2, 1, 2),
                    (2, 1, 3),
                    (2, 2, 0),
                    (2, 2, 4),
                    (2, 2, 5),
                    (2, 2, 15),
                    (3, 0, 2),
                    (3, 0, 13),
                    (3, 0, 14),
                    (3, -1, 0),
                    (3, 1, 2),
                    (3, 1, 3),
                    (4, 0, 3),
                    (4, -1, 2)
                ],
                (0, 0, 3): [
                    (-2, 3, 3),
                    (-1, 2, 5),
                    (0, 1, 4),
                    (0, 1, 5),
                    (0, 2, 6),
                    (0, 3, 1),
                    (0, 3, 7),
                    (1, 1, 1),
                    (1, 1, 2),
                    (1, 2, 1),
                    (1, 2, 2),
                    (1, 2, 3),
                    (1, 2, 4),
                    (1, 2, 5),
                    (2, 1, 1),
                    (2, 1, 2),
                    (2, 2, 0),
                    (2, 2, 4),
                    (3, 0, 0),
                    (3, 0, 1),
                    (3, 0, 14),
                    (3, 0, 15),
                    (3, 1, 2),
                    (3, 1, 3),
                    (3, 2, 4),
                    (4, 0, 2),
                    (4, -1, 0)
                ]
            }
        else:
            print("Num headings '{}' not allowed!".format(num_headings))
            exit(1)


if __name__ == '__main__':
    # for testing purposes
    theta = 0  # np.pi / 4
    p = Primitives(scale=8, initial_heading=theta, num_headings=16)
    p.view(theta=theta)
