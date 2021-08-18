import math
from typing import List, Tuple, Iterable

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
                 inf_stream: bool, map_figsize=(5, 10), sim_figsize=(10, 10), y_axis_limit=100):
        # init two fig and ax objects
        # the first is for plotting the updated costmap, node plot, swath, and path
        # the second is for plotting the simulated ship, polygons
        self.map_fig, ax = plt.subplots(1, 2, figsize=map_figsize, sharex='all', sharey='all')
        self.node_ax, self.map_ax = ax
        self.sim_fig, self.sim_ax = plt.subplots(figsize=sim_figsize)
        self.ax = [self.node_ax, self.map_ax, self.sim_ax]

        # set the axes limits for all plots
        for ax in self.ax:
            ax.axis([0, costmap.m, 0, y_axis_limit])
            ax.set_aspect('equal')

        # remove axes ticks and labels to speed up animation
        self.sim_ax.set_xlabel('')
        self.sim_ax.set_xticks([])
        self.sim_ax.set_ylabel('')
        self.sim_ax.set_yticks([])

        # plot the nodes that were expanded
        self.node_plot_image = self.node_ax.imshow(
            self.create_node_plot(nodes_expanded, shape=costmap.cost_map.shape), origin='lower'
        )

        # plot the costmap
        self.costmap_image = self.map_ax.imshow(
            costmap.cost_map, origin='lower'
        )

        # sample points along path
        full_path = get_points_on_path(
            path, prim.num_headings, ship.initial_heading, ship.turning_radius
        )
        # store all the points along dubins path for later use
        self.full_path = np.asarray(full_path)

        # show the path on both the map and sim plot
        self.path_line = [
            *self.map_ax.plot(self.full_path[0], self.full_path[1], 'g'),
            *self.sim_ax.plot(self.full_path[0], self.full_path[1], 'r')
        ]

        # plot the nodes along the path and the nodes added from the smoothing step
        self.nodes_line = [
            *self.map_ax.plot(*path_nodes, 'bx'),
            *self.map_ax.plot(*smoothing_nodes, 'gx')
        ]

        # add the patches for the ice
        self.obs_patches = []
        for obs in costmap.obstacles:
            self.obs_patches.append(
                self.sim_ax.add_patch(
                    patches.Polygon(obs['vertices'], True, fill=True)
                )
            )

        # get full swath
        full_swath, full_swath_cost, _ = swath.compute_swath_cost(
            costmap.cost_map, self.full_path, ship.vertices
        )
        swath_im = np.zeros(full_swath.shape + (4,))  # init RGBA array
        # fill in the RGB values
        swath_im[:] = colors.to_rgba('m')
        swath_im[:, :, 3] = full_swath  # set pixel transparency to 0 if pixel value is 0
        # plot the full swath
        self.swath_image = self.map_ax.imshow(swath_im, origin='lower', alpha=0.3)

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

        # set up a button for pausing
        self.paused = False
        button_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(button_ax, 'Pause/Start')
        self.bnext.on_clicked(self.toggle_pause)

        # create a patch for node plot
        patch_nodes = patches.Patch(color=self.node_plot_image.cmap(self.node_plot_image.norm(1)))
        # create a path for swath
        patch_swath = patches.Patch(color=colors.to_rgba('m'), alpha=0.6)
        # add legend
        self.map_fig.legend(
            (*self.nodes_line, self.path_line[0], patch_nodes, patch_swath),
            ('path nodes', 'smoothing nodes', 'path', 'expanded nodes', 'swath')  # , loc=(0.5, 0)
        )

        # add titles
        self.node_ax.set_title('Node plot')
        self.map_ax.set_title('Costmap')

        # add a textbox to display relevant metrics
        self.metrics = [len(nodes_expanded), full_swath_cost, 0, 0, 0]
        text_str = 'Number nodes: {}' \
                   '\nNew path\n  - cost: {:.0f}\n  - count: {}' \
                   '\nOld path\n  - cost: {:.0f}\n  - count: {}' \
            .format(*self.metrics)

        # place a text box in upper left in axes coords
        self.map_fig_text = self.map_fig.text(
            0.02, 0.985, text_str, fontsize=14, verticalalignment='top',
            bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
        )

    def toggle_pause(self, *args, **kwargs):
        print('Toggled pause:', not self.paused)
        self.paused = not self.paused

    def update_map(self, cost_map: np.ndarray) -> None:
        # update the costmap plot
        self.costmap_image.set_data(cost_map)

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

    def animate_ship(self, ship, horizon, move_yaxis_threshold) -> None:
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
        for poly, patch in zip(polygons, self.obs_patches):
            heading = poly.body.angle
            R = np.asarray([
                [math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]
            ])
            vs = np.asarray(poly.get_vertices()) @ R + np.asarray(poly.body.position)
            patch.set_xy(vs)

    def animate_map(self):
        # update y axis
        self.map_ax.set_ylim(self.sim_ax.get_ylim())

        # draw artists for map plot
        for artist in [*self.nodes_line, self.swath_image, self.path_line[0],
                       self.costmap_image, self.map_ax.yaxis]:
            self.map_ax.draw_artist(artist)

        # draw artists for node plot
        for artist in [self.node_plot_image, self.node_ax.yaxis]:
            self.node_ax.draw_artist(artist)

        self.map_fig.canvas.blit(self.map_fig.bbox)
        self.map_fig.canvas.flush_events()

    def animate_text(self, new_path_cost: float, new_path_count: int,
                     old_path_cost: float, old_path_count: Tuple, num_nodes: int = None):
        if num_nodes is None:
            num_nodes = self.metrics[0]

        text_str = 'Number nodes: {}' \
                   '\nNew path\n  - cost: {:.0f}\n  - count: {}'

        if old_path_cost is None:
            old_path_cost = 'Expired!'
            text_str += '\nOld path {}\n  - cost: N/A\n  - count: {}'
        else:
            text_str += '\nOld path\n  - cost: {:.0f}\n  - count: {}'

        self.metrics = [num_nodes, new_path_cost, new_path_count, old_path_cost, old_path_count]
        text_str = text_str.format(*self.metrics)

        # update text data
        self.map_fig_text.set_text(text_str)

        # update figure
        self.map_fig.draw_artist(self.map_fig_text)
        self.map_fig.canvas.blit(self.map_fig.bbox)
        self.map_fig.canvas.flush_events()

    def get_sim_artists(self) -> Iterable:
        # this is only useful when blit=True in FuncAnimation
        # which requires returning a list of artists that have changed in the sim fig
        return (
            self.path_line[1], self.ship_patch, *self.obs_patches,
            self.horizon_area, self.sim_ax.yaxis,
        )

    @staticmethod
    def create_node_plot(nodes_expanded: List, shape: Tuple) -> np.ndarray:
        node_plot = np.zeros(shape)
        for node in nodes_expanded:
            r, c = int(round(node[1])), int(round(node[0]))
            node_plot[r, c] = node_plot[r, c] + 1
        return node_plot
