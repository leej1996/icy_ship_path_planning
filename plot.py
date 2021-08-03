import math
from typing import List, Tuple, Iterable, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, colors
from pymunk import Poly

import swath
from cost_map import CostMap
from primitives import Primitives
from ship import Ship
from utils import get_points_on_path


class Plot:
    """
    Aggregates all plotting objects into a single class
    """
    def __init__(self, costmap: CostMap, prim: Primitives, ship: Ship, nodes_expanded: List,
                 path: List, path_nodes: Tuple[List, List], smoothing_nodes: Tuple[List, List], horizon: int,
                 inf_stream: bool, map_figsize=(5, 10), sim_figsize=(10, 10)):
        # init two fig and ax objects
        # the first is for plotting the updated costmap, node plot, and path
        # the second is for plotting the simulated ship and polygons
        self.map_fig, self.map_ax = plt.subplots(1, 2, figsize=map_figsize)
        self.sim_fig, self.sim_ax = plt.subplots(figsize=sim_figsize)

        # set the axes limits for plots
        for ax in self.map_ax:
            ax.axis([0, costmap.m, 0, costmap.n])
            ax.set_aspect("equal")
            # ax.yaxis.set_animated(True)
        self.sim_ax.axis([0, costmap.m, 0, costmap.n])
        self.sim_ax.set_aspect("equal")
        # self.sim_ax.yaxis.set_animated(True)

        # plot the nodes that were expanded
        self.node_plot_image = self.map_ax[1].imshow(
            self.create_node_plot(nodes_expanded, shape=costmap.cost_map.shape), origin='lower'
        )

        # plot the costmap
        self.costmap_image = self.map_ax[0].imshow(
            costmap.cost_map, origin='lower'
        )

        # sample points along path
        full_path = get_points_on_path(
            path, prim.num_headings, ship.initial_heading, ship.turning_radius
        )
        self.full_path = np.asarray(full_path)

        # show the path on both the map and sim plot
        self.path_line = [
            *self.map_ax[0].plot(self.full_path[0], self.full_path[1], 'g'),
            *self.sim_ax.plot(self.full_path[0], self.full_path[1], 'r')
        ]
        # store all the points along dubins path for later use

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

        # get full swath
        full_swath, *_ = swath.compute_swath_cost(
            costmap.cost_map, self.full_path, ship.vertices
        )
        swath_im = np.zeros(full_swath.shape + (4,))  # init RGBA array
        # fill in the RGB values
        swath_im[:] = colors.to_rgba('m')
        swath_im[:, :, 3] = full_swath  # set pixel transparency to 0 if pixel value is 0
        # plot the full swath
        self.swath_image = self.map_ax[0].imshow(swath_im, origin='lower', alpha=0.3)

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

        # display a lightly shaded region for the horizon
        if type(horizon) is int:
            self.horizon_area = self.sim_ax.fill_between(
                x=np.arange(0, costmap.m), y1=path[-1][1], y2=path[0][1], color='C1', alpha=0.3, zorder=0
            )

        self.inf_stream = inf_stream
        self.prev_ship_pos = ship.body.position

    def update_map(self, cost_map: np.ndarray, obstacles: Dict) -> None:
        # update the costmap plot
        self.costmap_image.set_data(cost_map)

        # update the patches on the map plot
        for idx, obs in enumerate(obstacles):
            # only add patch if obs is on the map
            if obs['on_map']:
                self.obs_patches[0][idx].set_xy(obs['vertices'])

        if self.inf_stream:
            pass  # TODO
            # left, right, bottom, top = self.costmap_image.get_extent()
            # self.costmap_image.set_extent(
            #     (left, right, self.prev_ship_pos[1] - 0.5, top + self.prev_ship_pos[1] + 0.5)
            # )

    def update_path(self, full_path: np.ndarray, full_swath: np.ndarray, path_nodes: Tuple[List, List],
                    smoothing_nodes: Tuple[List, List], nodes_expanded: List) -> None:
        self.full_path = full_path
        p_x, p_y, _ = self.full_path
        # show the new path on both the sim and map plot
        for line in self.path_line:
            line.set_data(p_x, p_y)

        # update the node plot
        self.node_plot_image.set_data(
            self.create_node_plot(nodes_expanded, shape=self.node_plot_image.get_array().shape)
        )

        # update the nodes lines
        for line, nodes in zip(self.nodes_line, [path_nodes, smoothing_nodes]):
            line.set_data(nodes[0], nodes[1])

        swath_im = np.zeros(full_swath.shape + (4,))  # init RGBA array
        # fill in the RGB values
        swath_im[:] = colors.to_rgba('m')
        swath_im[:, :, 3] = full_swath  # set pixel transparency to 0 if pixel value is 0
        # update the swath image
        self.swath_image.set_data(swath_im)

    def animate_ship(self, ship, horizon, move_yaxis_threshold=20) -> None:
        heading = ship.body.angle
        R = np.asarray([
            [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
        ])
        vs = np.asarray(ship.shape.get_vertices()) @ R + np.asarray(ship.body.position)
        self.ship_patch.set_xy(vs)

        # compute how much ship has moved in the y direction since last step
        offset = np.array([0, ship.body.position.y - self.prev_ship_pos[1]])
        # shade in the area of the map that is part of the horizon
        if type(horizon) is int:
            self.horizon_area.get_paths()[0].vertices += offset
            self.prev_ship_pos = ship.body.position  # update prev ship position

        # update y axis if necessary
        if self.inf_stream and ship.body.position.y > move_yaxis_threshold:
            ymin, ymax = self.sim_ax.get_ylim()
            self.sim_ax.set_ylim([ymin + offset[1], ymax + offset[1]])

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
            self.path_line[1], self.ship_patch, *self.obs_patches[1], self.horizon_area,
            self.map_ax[0].yaxis, self.map_ax[1].yaxis, self.sim_ax.yaxis, self.swath_image
        )

    @staticmethod
    def create_node_plot(nodes_expanded: List, shape: Tuple) -> np.ndarray:
        node_plot = np.zeros(shape)
        for node in nodes_expanded:
            r, c = int(round(node[1])), int(round(node[0]))
            node_plot[r, c] = node_plot[r, c] + 1
        return node_plot
