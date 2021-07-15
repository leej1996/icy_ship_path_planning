import math
import queue
import time
from multiprocessing import Process, Queue

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation

import swath
from a_star_search import AStar
from cost_map import CostMap
from primitives import Primitives
from ship import Ship
from state_lattice import snap_to_lattice
from utils import heading_to_world_frame, plot_path


def gen_path(state_q: Queue, path_q: Queue, initial_heading: object, swath_dict: swath.Swath):
    while True:  # TODO: better stopping condition
        try:
            new_state = state_q.get_nowait()
        except queue.Empty:
            print("waiting")
            time.sleep(1.0)
            continue

        # compute next state
        curr_pos = new_state
        print("\n################\nGot a new state!")
        print("Starting node", curr_pos)
        # compute the heading of current in the world frame
        initial_heading.v = heading_to_world_frame(curr_pos[2], initial_heading.v, num_headings)
        print("New initial_heading", initial_heading.v)
        # snap goal to lattice
        snapped_goal = snap_to_lattice(curr_pos, goal_pos, initial_heading.v, turning_radius, num_headings)
        new_start = (*curr_pos[:2], 0)  # straight ahead of boat is 0

        print("New start", new_start)
        print("New goal", snapped_goal)

        # the angle to rotate the primitives and swath keys based on the difference
        # between previous ship heading and current initial heading
        theta = initial_heading.v - ship.initial_heading
        print("Rotating primitives and swaths by", theta)

        # rotate primitives
        prim.rotate(theta)

        # update swath keys
        swath_dict = swath.update_swath(theta, swath_dict)

        # update ship initial heading
        ship.initial_heading = initial_heading.v

        # run search
        worked, new_path, nodes_visited, x1, y1, x2, y2, orig_path = \
            a_star.search(new_start, snapped_goal, swath_dict, smooth_path)

        if new_path != "Fail":
            new_path.reverse()

            if len(new_path) <= 1:
                print("break")
                break

            if path_q.empty():
                path_q.put([new_path, ship.initial_heading])

        time.sleep(2.0)


def update(frame, state_q: Queue, path_q: Queue):
    try:
        path, new_heading = path_q.get_nowait()
        path_obj.v = path
        path_obj.o = 0
        ship.initial_heading = new_heading
    except queue.Empty:
        print("no new path...")
        # if no new path just continue plotting along existing path
        path_obj.o += 1
        n_x, n_y, p_x, p_y, theta = plot_path(ax, path_obj.v, costmap_obj.cost_map, ship, num_headings, return_points=True)
        ln2.set_data(p_x[path_obj.o:], p_y[path_obj.o:])
        ln1.set_data([p_x[path_obj.o], *n_x[1:]], [p_y[path_obj.o], *n_y[1:]])

        if state_q.empty():
            state_q.put([p_x[20], p_y[20], theta[20]])
        time.sleep(0.5)

        return []

    print("Got a new path!", path)
    # update plot
    n_x, n_y, p_x, p_y, theta = plot_path(ax, path_obj.v, costmap_obj.cost_map, ship, num_headings, return_points=True)
    ln2.set_data(p_x, p_y)
    ln1.set_data(n_x, n_y)

    # set new state to be next node along path
    if state_q.empty():
        state_q.put([p_x[10], p_y[10], theta[10]])

    return []


if __name__ == '__main__':
    # Resolution is 10 m
    num_headings = 8
    n = 300
    m = 40
    class Object(object):
        pass
    initial_heading = Object()
    initial_heading.v = np.pi / 2
    density = 3
    turning_radius = 8  # 300 m turn radius
    ship_vertices = np.array([[0, 4],
                              [1, 3],
                              [1, -4],
                              [-1, -4],
                              [-1, 3]])
    obstacle_penalty = 0.3
    start_pos = (20, 34, 0)  # (x, y, theta), possible values for theta 0 - 7 measured from ships positive x axis
    goal_pos = (20, 282, 0)
    print("Initial heading", initial_heading.v,
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
    ship = Ship(ship_vertices, start_pos, initial_heading.v, turning_radius)
    # get the primitives
    prim = Primitives(scale=turning_radius, initial_heading=initial_heading.v, num_headings=num_headings)

    # generate swaths
    swath_dict = swath.generate_swath(ship, prim)

    # initialize a star object
    a_star = AStar(g_weight=0.2, h_weight=0.8, cmap=costmap_obj,
                   primitives=prim, ship=ship, first_initial_heading=initial_heading.v)
    worked, new_path, nodes_visited, x1, y1, x2, y2, orig_path = \
        a_star.search(start_pos, goal_pos, swath_dict, smooth_path)
    new_path.reverse()
    path_obj = Object()
    path_obj.v = new_path
    path_obj.o = 0

    # setup figure
    fig, ax = plt.subplots(1, figsize=(5, 10))
    _, ln1, ln2 = plot_path(ax, [], costmap_obj.cost_map, ship, num_headings)

    # setup queues
    state_q = Queue()  # queue to send state information to A*
    state_q.put(start_pos)
    path_q = Queue()  # queue tp send path information to controller

    print('\nStart process...')
    find_path = Process(target=gen_path, args=(state_q, path_q, initial_heading, swath_dict))
    find_path.start()

    ani = FuncAnimation(fig, update, blit=True, interval=20, fargs=(state_q, path_q,))
    plt.show()

    print("...done with process")
    find_path.join()
    print('Completed multiprocessing')
