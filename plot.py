import math
from typing import List, Tuple, Iterable, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from pymunk import Poly

from cost_map import CostMap
from primitives import Primitives
from ship import Ship
from utils import get_points_on_dubins_path


class Plot:
    """
    Aggregates all plotting objects into a single class
    """
    def __init__(self, costmap: CostMap, prim: Primitives, ship: Ship, nodes_expanded: List,
                 path: List, path_nodes: Tuple[List, List], smoothing_nodes: Tuple[List, List],
                 map_figsize=(5, 10), sim_figsize=(10, 10)):
        # init two fig and ax objects
        # the first is for plotting the updated costmap, node plot, and path
        # the second is for plotting the simulated ship and polygons
        self.map_fig, self.map_ax = plt.subplots(1, 2, figsize=map_figsize)
        self.sim_fig, self.sim_ax = plt.subplots(figsize=sim_figsize)

        # set the axes limits for sim plot
        self.sim_ax.set_xlim(0, costmap.m)
        self.sim_ax.set_ylim(0, costmap.n)
        self.sim_ax.set_aspect("equal")

        # plot the nodes that were expanded
        self.node_plot_image = self.map_ax[1].imshow(
            self.create_node_plot(nodes_expanded, shape=costmap.cost_map.shape), origin='lower'
        )

        # plot the costmap
        self.costmap_image = self.map_ax[0].imshow(
            costmap.cost_map, origin='lower'
        )

        # plot the path
        p_x, p_y, p_theta = self.get_points_on_path(path, prim.num_headings, ship.initial_heading, ship.turning_radius)
        # show the path on both the map and sim plot
        self.path_line = [
            *self.map_ax[0].plot(p_x, p_y, 'g'),
            *self.sim_ax.plot(p_x, p_y, 'r')
        ]
        # create an array to store all the points along dubins path for later use
        self.full_path = np.asarray([p_x, p_y, p_theta])

        # plot the nodes along the path and the nodes added from the smoothing step
        self.nodes_line = [
            *self.map_ax[0].plot(*path_nodes, 'bx'),
            *self.map_ax[0].plot(*smoothing_nodes, 'gx')
        ]

        # add the patches for the ice and ship to both plots
        self.obs_patches = [[], []]
        for obs in costmap.obstacles:
            self.obs_patches[0].append(
                self.map_ax[0].add_patch(
                    patches.Polygon(obs['vertices'], True, fill=False)
                )
            )
            self.obs_patches[1].append(
                self.sim_ax.add_patch(
                    patches.Polygon(obs['vertices'], True, fill=True)
                )
            )

        # create polygon patch for ship
        vs = np.zeros_like(np.asarray(ship.shape.get_vertices()))
        for i, ship_vertex in enumerate(ship.shape.get_vertices()):
            x, y = ship_vertex.rotated(ship.body.angle) + ship.body.position
            vs[i][0] = x
            vs[i][1] = y

        # add patch for ship
        self.ship_patch = self.sim_ax.add_patch(
            patches.Polygon(vs, True, color='green')
        )

    def update_map(self, cost_map: np.ndarray, obstacles: Dict) -> None:
        # update the costmap plot
        self.costmap_image.set_data(cost_map)

        # update the patches on the map plot
        for idx, obs in enumerate(obstacles):
            # only add patch if obs is on the map
            if obs['on_map']:
                self.obs_patches[0][idx].set_xy(obs['vertices'])

    def update_path(self, path: List, num_headings: int, initial_heading: float, turning_radius: float,
                    path_nodes: Tuple[List, List], smoothing_nodes: Tuple[List, List], nodes_expanded: List) -> None:
        # plot the new path
        p_x, p_y, p_theta = self.get_points_on_path(path, num_headings, initial_heading, turning_radius)
        # show the new path on both the sim and map plot
        for line in self.path_line:
            line.set_data(p_x, p_y)
        self.full_path = np.asarray([p_x, p_y, p_theta])

        # update the node plot
        self.node_plot_image.set_data(
            self.create_node_plot(nodes_expanded, shape=self.node_plot_image.get_array().shape)
        )

        # update the nodes lines
        for line, nodes in zip(self.nodes_line, [path_nodes, smoothing_nodes]):
            line.set_data(nodes[0], nodes[1])

    def animate_ship(self, ship) -> None:
        heading = ship.body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vs = np.asarray(ship.shape.get_vertices()) @ R + np.asarray(ship.body.position)
        self.ship_patch.set_xy(vs)

    def animate_obstacles(self, polygons: List[Poly]) -> None:
        for poly, patch in zip(polygons, self.obs_patches[1]):
            heading = poly.body.angle
            R = np.asarray([
                [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
            ])
            vs = np.asarray(poly.get_vertices()) @ R + np.asarray(poly.body.position)
            patch.set_xy(vs)

    def get_sim_artists(self) -> Iterable:
        return (
            self.path_line[1], self.ship_patch, *self.obs_patches[1]
        )

    @staticmethod
    def get_points_on_path(path: List, num_headings: int, initial_heading: float, turning_radius: float,
                           show_prims: bool = False, eps: float = 1e-5) -> Tuple[List, List, List]:
        p_x, p_y, p_theta = [], [], []
        # reverse the path
        path = path[::-1]
        for i in range(np.shape(path)[0] - 1):
            p1 = path[i]
            p2 = path[i + 1]
            x, y, theta = get_points_on_dubins_path(
                p1, p2, num_headings, initial_heading, turning_radius, eps
            )
            p_x.extend(x)
            p_y.extend(y)
            p_theta.extend(theta)

        return p_x, p_y, p_theta

    @staticmethod
    def create_node_plot(nodes_expanded: List, shape: Tuple) -> np.ndarray:
        node_plot = np.zeros(shape)
        for node in nodes_expanded:
            r, c = int(round(node[1])), int(round(node[0]))
            node_plot[r, c] = node_plot[r, c] + 1
        return node_plot
