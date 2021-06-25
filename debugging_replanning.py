import math
import os
import pickle
import time

import dubins
import numpy as np
import pymunk
import pymunk.constraints
from matplotlib import animation
from matplotlib import patches
from matplotlib import pyplot as plt
from pymunk.vec2d import Vec2d
from skimage import draw
from skimage import transform

from a_star_search import AStar
from cost_map import CostMap
from primitives import Primitives
from ship import Ship
from state_lattice import snap_to_lattice, generate_swath
from utils import heading_to_world_frame

# Resolution is 10 m
n = 300
m = 40
initial_heading = 3 * math.pi / 4
density = 3
turning_radius = 8  # 300 m turn radius
ship_vertices = np.array([[0, 4],
                          [1, 3],
                          [1, -4],
                          [-1, -4],
                          [-1, 3]])
obstacle_penalty = 3
start_pos = (20, 34, 0)  # (x, y, theta), possible values for theta 0 - 7 measured from ships positive x axis
goal_pos = snap_to_lattice(start_pos=start_pos, goal_pos=(20, 282, math.pi / 2 - initial_heading),
                           initial_heading=initial_heading, turning_radius=turning_radius)
print("GOAL", goal_pos)
smooth_path = False

# initialize costmap
costmap_obj = CostMap(n, m, obstacle_penalty)

# generate random obstacles
costmap_obj.generate_obstacles(start_pos, goal_pos, num_obs=130, min_r=1, max_r=8,
                               upper_offset=20, lower_offset=20, allow_overlap=False)

# initialize ship object
ship = Ship(ship_vertices, start_pos, goal_pos, initial_heading, turning_radius)
print("TURN RADIUS", ship.calc_turn_radius(45, 2))
# get the primitives
prim = Primitives(scale=turning_radius, initial_heading=initial_heading)

# generate swaths
ordinal_swaths = generate_swath(ship, prim.edge_set_ordinal, 1, prim)
cardinal_swaths = generate_swath(ship, prim.edge_set_cardinal, 0, prim)

# initialize a star object
a_star = AStar(g_weight=0.5, h_weight=0.5, cmap=costmap_obj,
               primitives=prim, ship=ship, first_initial_heading=initial_heading)

worked, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
    a_star.search(start_pos, goal_pos, cardinal_swaths, ordinal_swaths, smooth_path)

smoothed_edge_path.reverse()
print("\nOriginal path", smoothed_edge_path)
fig1, ax1 = plt.subplots(1, 2, figsize=(5, 10))

# run a star starting at each node in original found path
for node in smoothed_edge_path[1:]:
    print("\nStarting node", node)
    initial_heading = heading_to_world_frame(node[2], initial_heading)
    ship.initial_heading = initial_heading
    print("New initial_heading", initial_heading)
    snapped_goal = snap_to_lattice(node, goal_pos, initial_heading, turning_radius)

    new_start = (*node[:2], 0)  # straight ahead of boat is 0
    print("New start", new_start)
    print("New goal", snapped_goal)

    # save current edge sets
    prev_edge_set_ordinal = prim.edge_set_ordinal.copy()
    prev_edge_set_cardinal = prim.edge_set_cardinal.copy()

    # rotate primitives
    prim.rotate(initial_heading)

    # update swath keys
    new_ordinal_swaths = {}
    new_cardinal_swaths = {}
    for old_e, e in zip(prev_edge_set_ordinal, prim.edge_set_ordinal):
        for i in [1, 3, 5, 7]:
            new_ordinal_swaths[tuple(e), i] = ordinal_swaths[tuple(old_e), i]

    for old_e, e in zip(prev_edge_set_cardinal, prim.edge_set_cardinal):
        for i in [0, 2, 4, 6]:
            new_cardinal_swaths[tuple(e), i] = cardinal_swaths[tuple(old_e), i]

    worked, new_path, nodes_visited, x1, y1, x2, y2, orig_path = \
        a_star.search(new_start, snapped_goal, new_cardinal_swaths, new_ordinal_swaths)

    new_path.reverse()
    print("\nNew path", new_path)