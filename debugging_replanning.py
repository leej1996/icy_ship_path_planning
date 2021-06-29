import math

import numpy as np
from matplotlib import pyplot as plt

from a_star_search import AStar
from cost_map import CostMap
from primitives import Primitives
from ship import Ship
from state_lattice import snap_to_lattice, generate_swath
from utils import heading_to_world_frame, plot_path

if __name__ == '__main__':
    # Resolution is 10 m
    n = 300
    m = 40
    initial_heading = math.pi / 2
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
    print("Initial heading", initial_heading,
          "\nStart position", start_pos,
          "\nGoal position", goal_pos)

    # disables smoothing part
    smooth_path = False

    # initialize costmap
    costmap_obj = CostMap(n, m, obstacle_penalty)

    # generate random obstacles
    costmap_obj.generate_obstacles(start_pos, goal_pos, num_obs=130, min_r=1, max_r=8,
                                   upper_offset=20, lower_offset=20, allow_overlap=False)

    # initialize ship object
    ship = Ship(ship_vertices, start_pos, initial_heading, turning_radius)
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
    print("Original path", smoothed_edge_path)

    # initialize current node
    curr_pos = smoothed_edge_path[1]

    # initialize goal for stopping condition
    snapped_goal = goal_pos

    # run a star until goal node is exactly reached
    while (
            AStar.dist(curr_pos, snapped_goal) >= 1e-3
    ):
        print("\n################\nNEXT STEP\nStarting node", curr_pos)
        # compute the heading of current in the world frame
        initial_heading = heading_to_world_frame(curr_pos[2], initial_heading)
        print("New initial_heading", initial_heading)
        # snap goal to lattice
        snapped_goal = snap_to_lattice(curr_pos, goal_pos, initial_heading, turning_radius)
        new_start = (*curr_pos[:2], 0)  # straight ahead of boat is 0

        print("New start", new_start)
        print("New goal", snapped_goal)

        # the angle to rotate the primitives and swath keys based on the difference
        # between previous ship heading and current initial heading
        theta = initial_heading - ship.initial_heading
        print("Rotating primitives and swaths by", theta)

        # rotate primitives
        prim.rotate(theta)

        # update swath keys
        ordinal_swaths, cardinal_swaths = prim.update_swath(theta=theta,
                                                            ord_swath=ordinal_swaths,
                                                            card_swath=cardinal_swaths)
        # update ship initial heading
        ship.initial_heading = initial_heading

        # run search
        worked, new_path, nodes_visited, x1, y1, x2, y2, orig_path = \
            a_star.search(new_start, snapped_goal, cardinal_swaths, ordinal_swaths, smooth_path=False)

        if new_path != "Fail":
            new_path.reverse()
            print("\nNew path", new_path)

            if len(new_path) <= 1:
                break
            curr_pos = new_path[1]

            fig, ax = plt.subplots(1, figsize=(5, 10))
            plot_path(ax, new_path, costmap_obj.cost_map, ship)
            plt.show()
        else:
            exit(1)

    print("\n\nDONE!!")
