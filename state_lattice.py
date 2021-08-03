import math
import os
import pickle
import random
import time
from multiprocessing import Process, Pipe, Event, Queue
from queue import Empty

import numpy as np
import pymunk
import pymunk.constraints
from matplotlib import animation
from matplotlib import pyplot as plt
from pymunk.vec2d import Vec2d
from simple_pid import PID

import swath
from a_star_search import AStar, gen_path
from cost_map import CostMap
from plot import Plot
from primitives import Primitives
from pure_pursuit import TargetCourse, State
from ship import Ship
from utils import create_polygon, Path

random.seed(1)  # make the simulation the same each time, easier to debug
at_goal = False


def state_lattice_planner(n: int, m: int, file_name: str = "test", g_weight: float = 0.5, h_weight: float = 0.5,
                          costmap_file: str = "",
                          start_pos: tuple = (20, 10, 0), goal_pos: tuple = (20, 280, 0),
                          initial_heading: float = math.pi / 2, padding: int = 0,
                          turning_radius: int = 8, vel: int = 10, num_headings: int = 8,
                          num_obs: int = 130, min_r: int = 1, max_r: int = 8, upper_offset: int = 20,
                          lower_offset: int = 20, allow_overlap: bool = False,
                          obstacle_density: int = 6, obstacle_penalty: float = 3,
                          Kp: float = 3, Ki: float = 0.08, Kd: float = 0.5, inf_stream: bool = False,
                          save_animation: bool = False, save_costmap: bool = False, smooth_path: bool = False,
                          replan: bool = False, horizon: int = np.inf):
    # PARAM SETUP
    # --- costmap --- #
    load_costmap_file = costmap_file
    ship_vertices = np.array([[-1, -4],
                              [1, -4],
                              [1, 2],
                              [0, 4],
                              [-1, 2]])

    # load costmap object from file if specified
    if load_costmap_file:
        with open(load_costmap_file, "rb") as fd:
            costmap_obj = pickle.load(fd)
            # recompute costmap costs if obstacle penalty is different than original
            if costmap_obj.obstacle_penalty != obstacle_penalty:
                costmap_obj.update2(obstacle_penalty)
    else:
        # initialize costmap
        costmap_obj = CostMap(n, m, obstacle_penalty, inf_stream)

        # generate random obstacles
        costmap_obj.generate_obstacles(start_pos[1], goal_pos[1], num_obs, min_r, max_r,
                                       upper_offset, lower_offset, allow_overlap)

    orig_obstacles = costmap_obj.obstacles.copy()

    # initialize ship object
    ship = Ship(ship_vertices, start_pos, initial_heading, turning_radius, padding)

    # get the primitives
    prim = Primitives(turning_radius, initial_heading, num_headings)

    # generate swath dict
    swath_dict = swath.generate_swath(ship, prim)

    print("WEIGHTS", g_weight, h_weight)
    # initialize a star object
    a_star = AStar(g_weight, h_weight, cmap=costmap_obj,
                   primitives=prim, ship=ship, first_initial_heading=initial_heading)

    # compute current goal
    curr_goal = (goal_pos[0], min(goal_pos[1], (start_pos[1] + horizon)), goal_pos[2])

    t0 = time.clock()
    worked, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
        a_star.search(start_pos, curr_goal, swath_dict, smooth_path)

    init_plan_time = time.clock() - t0
    print("Time elapsed: ", init_plan_time)
    print("Hz", 1 / init_plan_time)

    if worked:
        plot_obj = Plot(
            costmap_obj, prim, ship, nodes_visited, smoothed_edge_path.copy(),
            path_nodes=(x1, y1), smoothing_nodes=(x2, y2), horizon=horizon, inf_stream=inf_stream
        )
        path = Path(plot_obj.full_path)
    else:
        print("Failed to find path at step 0")
        exit(1)

    # init pymunk sim
    space = pymunk.Space()
    space.add(ship.body, ship.shape)
    space.gravity = (0, 0)
    staticBody = space.static_body  # create a static body for friction constraints

    # create the pymunk objects and the polygon patches for the ice
    polygons = []
    for obs in costmap_obj.obstacles:
        polygons.append(
            create_polygon(
                space, staticBody, (obs['vertices'] - np.array(obs['centre'])).tolist(),
                *obs['centre'], density=obstacle_density
            )
        )

    # From pure pursuit
    state = State(x=start_pos[0], y=start_pos[1], yaw=0.0, v=0.0)
    target_course = TargetCourse(path.path[0], path.path[1])
    target_ind = target_course.search_target_index(state)

    # init PID controller
    pid = PID(Kp, Ki, Kd, 0)
    pid.output_limits = (-1, 1)  # limit on PID output

    # generator to end matplotlib animation when it reaches the goal
    def gen():
        global at_goal
        i = 0
        while not at_goal:
            i += 1
            yield i
        raise StopIteration  # should stop animation

    def animate(frame, queue_state, pipe_path):
        global at_goal

        steps = 10
        # move simulation forward 20 ms seconds:
        for x in range(steps):
            space.step(0.02 / steps)

        # get current state
        ship_pos = (ship.body.position.x, ship.body.position.y, 0)  # straight ahead of boat is 0

        # check if ship has made it past the goal line
        if ship.body.position.y >= goal_pos[1]:
            at_goal = True
            print("\nAt goal, shutting down...")
            plt.close(plot_obj.map_fig)
            plt.close(plot_obj.sim_fig)
            queue_state.close()
            shutdown_event.set()
            return []

        # Pymunk takes left turn as negative and right turn as positive in ship.body.angle
        # To get proper error, we must flip the sign on the angle, as to calculate the setpoint,
        # we look at a point one lookahead distance ahead, and find the angle to that point with
        # arctan2, but using this, we will get positive values on the left and negative values on the right
        # As the angular velocity in pymunk uses the same convention as ship.body.angle, we must flip the sign
        # of the output as well
        output = -pid(-ship.body.angle)

        # should play around with frequency at which new state data is sent
        if frame % 20 == 0 and frame != 0 and replan:
            # update costmap
            costmap_obj.update(polygons)
            try:
                # empty queue to ensure latest state data is pushed
                queue_state.get_nowait()
            except Empty:
                pass

            # send updated state via queue
            queue_state.put({
                'ship_pos': ship_pos,
                'ship_body_angle': ship.body.angle,
                'costmap': costmap_obj.cost_map,
                'obstacles': costmap_obj.obstacles,
            }, block=False)

        # check if there is a new path
        if pipe_path.poll():
            # get new path
            path_data = pipe_path.recv()
            new_path = path_data['path']  # this is the full path and in the correct order i.e. start -> goal
            print('\nReceived replanned path!')

            # compute swath cost of new path up until the max y distance of old path for a fair comparison
            full_swath, full_cost, current_cost = swath.compute_swath_cost(
                costmap_obj.cost_map, new_path, ship.vertices, threshold_dist=path.path[1][-1]
            )
            try:
                assert full_cost >= current_cost  # sanity check
            except AssertionError:
                print("Full and partial swath costs", full_cost, current_cost)

            path_expired = False
            prev_cost = None

            # check if old path is 'expired' regardless of costs
            if (path.path[1][-1] - ship_pos[1]) < horizon / 2:
                path_expired = True

            else:
                # clip old path based on ship y position
                old_path = path.clip_path(ship_pos[1])

                # compute cost of clipped old path
                _, prev_cost, _ = swath.compute_swath_cost(costmap_obj.cost_map, old_path, ship.vertices)

                print('\nPrevious Cost: {prev_cost:.3f}'.format(prev_cost=prev_cost))
                print('Current Cost: {current_cost:.3f}\n'.format(current_cost=current_cost))

            if path_expired or current_cost < prev_cost:
                if path_expired:
                    print("Path expired, applying new path regardless of cost!")
                else:
                    print("New path better than old path!")
                    path.new_path_cnt += 1

                plot_obj.update_path(
                    new_path, full_swath, path_data['path_nodes'],
                    path_data['smoothing_nodes'], path_data['nodes_expanded']
                )

                # update to new path
                path.path = new_path
                ship.set_path_pos(0)

                # update pure pursuit objects with new path
                target_course.update(path.path[0], path.path[1])
                state.update(ship.body.position.x, ship.body.position.y, ship.body.angle)

                # update costmap and map fig
                plot_obj.update_map(costmap_obj.cost_map, costmap_obj.obstacles)
                plot_obj.map_fig.canvas.draw()

            else:
                print("Old path better than new path")
                path.old_path_cnt += 1

        if ship.path_pos < np.shape(path.path)[1] - 1:
            # Translate linear velocity into direction of ship
            x_vel = math.sin(ship.body.angle)
            y_vel = math.cos(ship.body.angle)
            mag = math.sqrt(x_vel ** 2 + y_vel ** 2)
            x_vel = x_vel / mag * vel
            y_vel = y_vel / mag * vel
            ship.body.velocity = Vec2d(x_vel, y_vel)

            # Assign output of PID controller to angular velocity
            ship.body.angular_velocity = output

            # Update the pure pursuit state
            state.update(ship.body.position.x, ship.body.position.y, ship.body.angle)

            # Get look ahead index
            ind = target_course.search_target_index(state)

            if ind != ship.path_pos:
                # Find heading from current position to look ahead point
                ship.set_path_pos(ind)
                dy = path.path[1][ind] - ship.body.position.y
                dx = path.path[0][ind] - ship.body.position.x
                angle = np.arctan2(dy, dx) - a_star.first_initial_heading
                # set setpoint for PID controller
                pid.setpoint = angle

        # at each step animate ship and obstacle patches
        plot_obj.animate_ship(ship, horizon)
        plot_obj.animate_obstacles(polygons)

        return plot_obj.get_sim_artists()

    # multiprocessing setup
    lifo_queue = Queue(maxsize=1)  # LIFO queue to send state information to A*
    conn_recv, conn_send = Pipe(duplex=False)  # pipe to send new path to controller and for plotting
    shutdown_event = Event()

    # setup a process to run A*
    print('\nStart process...')
    gen_path_process = Process(
        target=gen_path, args=(lifo_queue, conn_send, shutdown_event, ship, prim,
                               costmap_obj, swath_dict, a_star, goal_pos, horizon, smooth_path)
    )
    gen_path_process.start()

    # start animation in main process
    anim = animation.FuncAnimation(plot_obj.sim_fig,
                                   animate,
                                   frames=gen,
                                   fargs=(lifo_queue, conn_recv,),
                                   interval=20,
                                   blit=False,
                                   repeat=False,
                                   save_count=1500,
                                   )

    if save_animation:
        anim.save(os.path.join('gifs', file_name), writer=animation.PillowWriter(fps=30))
    plt.show()

    total_dist_moved = 0
    # Compare cost maps
    for i, j in zip(orig_obstacles, polygons):
        pos = (j.body.position.x, j.body.position.y)
        area = j.area
        if area > 4:
            total_dist_moved = a_star.dist(i['centre'], pos) * (area/2) + total_dist_moved

    print('TOTAL DIST MOVED', total_dist_moved)
    print('Old/new path counts', '\n\told path', path.old_path_cnt, '\n\tnew path', path.new_path_cnt)

    shutdown_event.set()
    print('\n...done with process')
    gen_path_process.join()
    print('Completed multiprocessing')

    # get response from user for saving costmap
    if save_costmap:
        costmap_obj.save_to_disk()
    return total_dist_moved, init_plan_time


def main():
    # PARAM SETUP
    # --- costmap --- #
    n = 100  # channel height
    m = 40  # channel width
    load_costmap_file = "" # "sample_costmaps/gold_test.pk"  # "sample_costmaps/random_obstacles_1.pk"

    # --- ship --- #
    start_pos = (20, 5, 0)  # (x, y, theta)
    goal_pos = (20, 140, 0)
    initial_heading = math.pi / 2
    turning_radius = 8
    vel = 10  # constant linear velocity of ship
    padding = 0  # padding around ship vertices to increase footprint when computing path costs

    # --- primitives --- #
    num_headings = 16

    # --- ice --- #
    num_obs = 100  # number of random ice obstacles
    min_r = 1  # min ice radius
    max_r = 5
    upper_offset = 20  # offset from top of costmap where ice stops
    lower_offset = 20  # offset from bottom of costmap where ice stops
    allow_overlap = False  # if True allow overlap in ice obstacles
    obstacle_density = 6
    obstacle_penalty = 2

    # --- A* --- #
    g_weight = 0.3  # cost = g_weight * g_score + h_weight * h_score
    h_weight = 0.7
    horizon = 50  # in metres
    smooth_path = False  # if True run smoothing algorithm as a post processing step
    replan = True  # if True rerun A* search at each time step

    # --- pid --- #
    Kp = 3
    Ki = 0.08
    Kd = 0.5

    # -- animation -- #
    inf_stream = True  # if True then simulation will run forever
    save_animation = False  # if True save animation and don't show it\
    save_costmap = False
    file_name = "test-1.gif"

    # automatic changes to params
    if inf_stream:
        upper_offset = -40

    state_lattice_planner(n, m, file_name=file_name, g_weight=g_weight, h_weight=h_weight,
                          costmap_file=load_costmap_file,
                          start_pos=start_pos, goal_pos=goal_pos, initial_heading=initial_heading, padding=padding,
                          turning_radius=turning_radius, vel=vel, num_headings=num_headings,
                          num_obs=num_obs, min_r=min_r, max_r=max_r, upper_offset=upper_offset,
                          lower_offset=lower_offset, allow_overlap=allow_overlap, obstacle_density=obstacle_density,
                          obstacle_penalty=obstacle_penalty, Kp=Kp, Ki=Ki, Kd=Kd,
                          save_animation=save_animation, save_costmap=save_costmap, smooth_path=smooth_path, replan=replan,
                          horizon=horizon, inf_stream=inf_stream)


if __name__ == "__main__":
    main()


# TODO:
# - delete poly objects that are not on costmap anymore
# - use loggers
# FIXME:
# - issue with smoothing
# - pymunk interactions