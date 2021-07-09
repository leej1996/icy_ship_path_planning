from typing import Tuple, Dict

import dubins
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import draw

from primitives import Primitives
from ship import Ship
from utils import heading_to_world_frame

Swath = Dict[Tuple, np.ndarray]


def generate_swath(ship: Ship, prim: Primitives, eps=1e-10) -> Swath:
    """
    Will have key of (edge, start heading)
    """
    swath_dict = {}
    for origin, edge_set in prim.edge_set_dict.items():
        start_pos = [prim.max_prim + ship.max_ship_length // 2] * 2 + [origin[2]]

        for e in edge_set:
            e = tuple(e)
            array = np.zeros([(prim.max_prim + ship.max_ship_length // 2) * 2 + 1] * 2, dtype=bool)
            translated_e = np.asarray(e) + np.array([start_pos[0], start_pos[1], 0])

            theta_0 = heading_to_world_frame(start_pos[2], ship.initial_heading, prim.num_headings)  # FIXME
            theta_1 = heading_to_world_frame(translated_e[2], ship.initial_heading, prim.num_headings)
            dubins_path = dubins.shortest_path((start_pos[0], start_pos[1], theta_0),
                                               (translated_e[0], translated_e[1], theta_1),
                                               ship.turning_radius - eps)

            configurations, _ = dubins_path.sample_many(0.1)

            for config in configurations:
                x_cell = int(round(config[0]))
                y_cell = int(round(config[1]))
                theta = config[2] - ship.initial_heading
                R = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                rot_vi = np.round(np.array([[x_cell], [y_cell]]) + R @ ship.vertices.T).astype(int)

                rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                array[rr, cc] = True

            # for each starting heading rotate the swath 4 times
            for i, h in enumerate(range(0, prim.num_headings, prim.num_headings // 4)):
                swath_dict[e, h + origin[2]] = np.rot90(array, k=i, axes=(1, 0))

    return swath_dict


def update_swath(theta: float, swath_dict: Swath) -> Swath:
    R = np.asarray([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # generate new keys for swath
    return {
        (tuple(k[0] @ R.T), k[1]): v for k, v in swath_dict.items()
    }


def view_swath(swath_dict: Swath, key: Tuple = None) -> None:
    if key is None:
        # get a random key from swath dict
        idx = np.random.randint(0, len(swath_dict), 1)[0]
        key = list(swath_dict.keys())[idx]
        print(key)
    plt.imshow(swath_dict[key], origin='lower')
    plt.show()
