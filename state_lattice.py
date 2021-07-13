import math
import pickle
import time

import numpy as np
import pymunk
import pymunk.constraints
from matplotlib import animation
from matplotlib import patches
from matplotlib import pyplot as plt
from pymunk.vec2d import Vec2d
from simple_pid import PID

import swath
from a_star_search import AStar
from cost_map import CostMap
from primitives import Primitives
from pure_pursuit import TargetCourse, State
from ship import Ship
from utils import heading_to_world_frame, get_points_on_dubins_path

at_goal = False


class Path:
    def __init__(self, path: np.array):
        self.path = path


def snap_to_lattice(start_pos, goal_pos, initial_heading, turning_radius, num_headings,
                    abs_init_heading=None, abs_goal_heading=None):
    # compute the spacing between base headings
    spacing = 2 * math.pi / num_headings

    # Rotate goal to lattice coordinate system
    R = np.asarray([
        [np.cos(initial_heading), -np.sin(initial_heading)],
        [np.sin(initial_heading), np.cos(initial_heading)]
    ])

    # determine how far from lattice the goal position is
    difference = R @ np.array([[goal_pos[0] - start_pos[0]], [goal_pos[1] - start_pos[1]]])
    diff_y = difference[1][0] % turning_radius
    diff_x = difference[0][0] % turning_radius

    # determine difference in heading
    abs_init_heading = heading_to_world_frame(start_pos[2], initial_heading, num_headings) \
        if abs_init_heading is None else abs_init_heading
    abs_goal_heading = heading_to_world_frame(goal_pos[2], initial_heading, num_headings) \
        if abs_goal_heading is None else abs_goal_heading
    diff = abs_goal_heading - abs_init_heading

    if diff < 0:
        diff = diff + (2 * math.pi)

    # check if x,y coordinates or heading are off lattice
    if diff_y != 0 or diff_x != 0 or diff % spacing != 0:
        if diff_y >= turning_radius / 2:
            new_goal_y = difference[1][0] + turning_radius - diff_y
        elif diff_y == 0:
            new_goal_y = difference[1][0]  # no change
        else:
            new_goal_y = difference[1][0] - diff_y

        if diff_x >= turning_radius / 2:
            new_goal_x = difference[0][0] + turning_radius - diff_x
        elif diff_x == 0:
            new_goal_x = difference[0][0]
        else:
            new_goal_x = difference[0][0] - diff_x

        # round to nearest cardinal/ordinal direction
        new_theta = round(diff / spacing)
        if new_theta > num_headings - 1:
            new_theta = new_theta - num_headings

        # rotate coordinates back to original frame
        new_goal = np.array([[new_goal_x], [new_goal_y]])
        new_goal = R.T @ new_goal
        goal_pos = (
            round(new_goal[0][0] + start_pos[0], 5),
            round(new_goal[1][0] + start_pos[1], 5),
            new_theta
        )

    return goal_pos


def create_polygon(space, staticBody, vertices, x, y, density):
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Poly(body, vertices)
    shape.density = density
    space.add(body, shape)

    # create pivot constraint to simulate linear friction
    pivot = pymunk.constraints.PivotJoint(staticBody, body, (0, 0))
    pivot.max_bias = 0
    pivot.max_force = 10000.0

    # create gear constraint to simulate angular friction
    gear = pymunk.constraints.GearJoint(staticBody, body, 0, 1)
    gear.max_bias = 0
    gear.max_force = 5000.0
    space.add(pivot, gear)
    return shape


def plot_path(fig1, costmap_obj, smoothed_edge_path, initial_heading, turning_radius, smooth_path, prim, x1, x2, y1, y2,
              nodes_visited, eps=1e0):
    plt.close(fig1)
    fig1, ax1 = plt.subplots(1, 2, figsize=(5, 10))
    ax1[0].imshow(costmap_obj.cost_map, origin='lower')
    PATH = smoothed_edge_path[::-1]
    path = np.zeros((3, 1))  # what is this used for?

    for i in range(np.shape(PATH)[0] - 1):
        P1 = PATH[i]
        P2 = PATH[i + 1]
        x, y, theta = get_points_on_dubins_path(P1, P2, prim.num_headings, initial_heading, turning_radius, eps)

        if not smooth_path and False:  # only want to show primitives on un smoothed path
            # find the base heading (e.g. cardinal or ordinal)
            num_base_h = prim.num_headings // 4
            arr = np.asarray([(P1[2] + num_base_h - h[2]) % num_base_h for h in prim.edge_set_dict.keys()])
            base_heading = np.argwhere(arr == 0)[0, 0]

            # get the edge set based on the current node heading
            edge_set = prim.edge_set_dict[(0, 0, base_heading)]

            for e in edge_set:
                p2 = AStar.concat(P1, e, base_heading, prim.num_headings)
                x3, y3, _ = get_points_on_dubins_path(P1, p2, prim.num_headings, initial_heading, turning_radius, eps)
                ax1[0].plot(x3, y3, 'r')

        ax1[0].plot(x, y, 'g')
        path = np.append(path, np.array([np.asarray(x).T, np.asarray(y).T, np.asarray(theta).T]), axis=1)

    path = np.delete(path, 0, 1)

    for obs in costmap_obj.obstacles:
        ax1[0].add_patch(patches.Polygon(obs['vertices'], True, fill=False))
    ax1[0].plot(x1, y1, 'bx')
    ax1[0].plot(x2, y2, 'gx')

    node_plot = create_node_plot(costmap_obj.n, costmap_obj.m, nodes_visited)
    ax1[1].imshow(node_plot, origin='lower')
    return fig1, path


def create_node_plot(n, m, nodes_visited):
    node_plot = np.zeros((n, m))
    for node in nodes_visited:
        r, c = int(round(node[1])), int(round(node[0]))
        node_plot[r, c] = node_plot[r, c] + 1
    return node_plot


def state_lattice_planner(file_name: str = "test", g_weight: float = 0.5, h_weight: float = 0.5, costmap_file: str = "",
                          start_pos: tuple = (20, 10, 0), goal_pos: tuple = (20, 280, 0),
                          initial_heading: float = math.pi / 2, padding: int = 0,
                          turning_radius: int = 8, vel: int = 10, num_headings: int = 8,
                          num_obs: int = 130, min_r: int = 1, max_r: int = 8, upper_offset: int = 20,
                          lower_offset: int = 20, allow_overlap: bool = False,
                          obstacle_density: int = 6, obstacle_penalty: float = 3,
                          Kp: float = 3, Ki: float = 0.08, Kd: float = 0.5,
                          save_animation: bool = False, smooth_path: bool = False, replan: bool = False):
    # PARAM SETUP
    # --- costmap --- #
    n = 300  # channel height
    m = 40  # channel width
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
        costmap_obj = CostMap(n, m, obstacle_penalty)

        # generate random obstacles
        costmap_obj.generate_obstacles(start_pos, goal_pos, num_obs, min_r, max_r,
                                       upper_offset, lower_offset, allow_overlap)

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

    t0 = time.clock()
    worked, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
        a_star.search(start_pos, goal_pos, swath_dict, smooth_path)

    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    print("Hz", 1 / t1)
    # print("smoothed path", smoothed_edge_path)
    print("NODES VISITED", len(nodes_visited))

    recomputed_original_cost, og_length = costmap_obj.compute_path_cost(path=orig_path, ship=ship,
                                                                        num_headings=prim.num_headings,
                                                                        reverse_path=True)
    smoothed_cost, smooth_length = costmap_obj.compute_path_cost(path=smoothed_edge_path.copy(), ship=ship,
                                                                 num_headings=prim.num_headings, reverse_path=True)
    straight_path_cost, straight_length = costmap_obj.compute_path_cost(path=[start_pos, goal_pos],
                                                                        num_headings=prim.num_headings, ship=ship)
    print("\nPath cost:"
          "\n\toriginal path:  {:.4f}"
          "\n\twith smoothing: {:.4f}"
          "\n\tstraight path:  {:.4f}\n".format(recomputed_original_cost, smoothed_cost, straight_path_cost))
    print("\nPath length:"
          "\n\toriginal path:  {:.4f}"
          "\n\twith smoothing: {:.4f}"
          "\n\tstraight path:  {:.4f}\n".format(og_length, smooth_length, straight_length))
    # try:
    #     assert smoothed_cost <= recomputed_original_cost <= straight_path_cost, \
    #         "smoothed cost should be less than original cost and original cost should be less than straight cost"
    # except AssertionError as error:
    #     print(error)
    #     costmap_obj.save_to_disk()

    fig1, ax1 = plt.subplots(1, 2, figsize=(5, 10))

    # '''
    # FIXME: why regenerate again, can't we just do this in the smoothing step????
    if worked:
        fig1, path_list = plot_path(fig1, costmap_obj, smoothed_edge_path, initial_heading, turning_radius,
                                    smooth_path, prim, x1, x2, y1, y2, nodes_visited)
    else:
        path = 0

    space = pymunk.Space()
    space.add(ship.body, ship.shape)
    space.gravity = (0, 0)
    staticBody = space.static_body  # create a static body for friction constraints

    polygons = []
    patch_list = []

    print("HEADING", ship.body.angle)
    i = 0
    vs = np.zeros_like(np.asarray(ship.shape.get_vertices()))
    for ship_vertex in ship.shape.get_vertices():
        x, y = ship_vertex.rotated(ship.body.angle) + ship.body.position
        vs[i][0] = x
        vs[i][1] = y
        i += 1

    ship_patch = patches.Polygon(vs, True, color='green')

    # print("GENERATE OBSTACLES")
    for obs in costmap_obj.obstacles:
        polygons.append(
            create_polygon(
                space, staticBody, (obs['vertices'] - np.array(obs['centre'])).tolist(),
                *obs['centre'], density=obstacle_density
            )
        )
        patch_list.append(patches.Polygon(obs['vertices'], True))

    path_list = path_list.T

    path = Path(path_list)

    # with open('test1.csv', 'a') as f:
    #    string = str(g_weight) + "," + str(h_weight) + "," + str(t1) + "," + str(1/t1) + "," + str(len(nodes_visited))
    #    print(string, file=f)
    # From pure pursuit
    state = State(x=start_pos[0], y=start_pos[1], yaw=0.0, v=0.0)
    target_course = TargetCourse(path.path.T[0], path.path.T[1])
    target_ind = target_course.search_target_index(state)

    fig2 = plt.figure()
    ax2 = plt.axes(xlim=(0, m), ylim=(0, n))
    ax2.set_aspect("equal")
    line, = ax2.plot(path.path.T[0], path.path.T[1], 'r')

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

    def init():
        # Initialize the matplotlib animation
        ax2.add_patch(ship_patch)
        line.set_ydata(path.path.T[1])
        line.set_xdata(path.path.T[0])
        for patch in patch_list:
            ax2.add_patch(patch)
        if not replan:
            ax2.plot(path.path.T[0], path.path.T[1], 'r')
        return []

    def animate(dt, ship_patch, ship, polygons, patch_list, path, fig1, swath_dict):
        global at_goal
        # print(dt)
        # 20 ms step size
        for x in range(10):
            space.step(2 / 100 / 10)

        ship_pos = (ship.body.position.x, ship.body.position.y, 0)  # straight ahead of boat is 0

        # Pymunk takes left turn as negative and right turn as positive in ship.body.angle
        # To get proper error, we must flip the sign on the angle, as to calculate the setpoint,
        # we look at a point one lookahead distance ahead, and find the angle to that point with
        # arctan2, but using this, we will get positive values on the left and negative values on the right
        # As the angular velocity in pymunk uses the same convention as ship.body.angle, we must flip the sign
        # of the output as well
        output = -pid(-ship.body.angle)

        # check if ship is at goal
        if a_star.dist(ship_pos, goal_pos) < 5:
            at_goal = True
        else:
            at_goal = False  # might not be needed

        if dt % 50 == 0 and dt != 0 and replan:
            print("\nNEXT STEP")
            # get heading of ship and rotate primitives/goal accordingly to new lattice
            ship.initial_heading = -ship.body.angle + a_star.first_initial_heading
            snapped_goal = snap_to_lattice(ship_pos, goal_pos, ship.initial_heading, turning_radius, prim.num_headings,
                                           abs_init_heading=ship.initial_heading)

            prim.rotate(-ship.body.angle, orig=True)

            swath_dict = swath.update_swath(theta=-ship.body.angle, swath_dict=swath_dict)

            print("INITIAL HEADING", ship.initial_heading)
            print("NEW GOAL", snapped_goal)
            print("NEW START", ship_pos)

            # Replan
            t0 = time.clock()
            worked, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
                a_star.search(ship_pos, snapped_goal, swath_dict, smooth_path)
            t1 = time.clock() - t0
            print("PLAN TIME", t1)

            if worked:
                print("Replanned Path", smoothed_edge_path)
                # update obstacles and generate new path from output of A*
                costmap_obj.update(polygons)
                fig1, path_list = plot_path(fig1, costmap_obj, smoothed_edge_path, ship.initial_heading, turning_radius,
                                            smooth_path, prim, x1, x2, y1, y2, nodes_visited)
                # plt.show()

                # update to new path
                path_list = path_list.T
                path.path = path_list
                line.set_xdata(path.path.T[0])
                line.set_ydata(path.path.T[1])
                ship.set_path_pos(0)

                # update pure pursuit objects with new path
                target_course.update(path.path.T[0], path.path.T[1])
                state.update(ship.body.position.x, ship.body.position.y, ship.body.angle)
                # plt.show(block=False)

        if ship.path_pos < np.shape(path.path)[0] - 1:
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
                dy = path.path[ind][1] - ship.body.position.y
                dx = path.path[ind][0] - ship.body.position.x
                angle = np.arctan2(dy, dx) - a_star.first_initial_heading
                # set setpoint for PID controller
                pid.setpoint = angle

        animate_ship(dt, ship, ship_patch)
        for poly, patch in zip(polygons, patch_list):
            animate_obstacle(dt, poly, patch)
        return []

    def animate_ship(dt, ship, patch):
        # update ship patch in matplotlib animation
        heading = ship.body.angle
        R = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        vs = np.asarray(ship.shape.get_vertices()) @ R + np.asarray(ship.body.position)
        patch.set_xy(vs)
        return patch,

    def animate_obstacle(dt, polygon, patch):
        # update obstacles in matplotlib animation
        heading = polygon.body.angle
        R = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        vs = np.asarray(polygon.get_vertices()) @ R + np.asarray(polygon.body.position)
        patch.set_xy(vs)
        return patch_list

    # print("START ANIMATION")
    # frames = np.shape(path.path)[0]
    anim = animation.FuncAnimation(fig2,
                                   animate,
                                   init_func=init,
                                   frames=gen,
                                   fargs=(ship_patch, ship, polygons, patch_list, path,
                                          fig1, swath_dict,),
                                   interval=20,
                                   blit=True,
                                   repeat=False,
                                   save_count=1500
                                   )

    if save_animation:
        file_name = "gifs/" + file_name
        anim.save(file_name, writer=animation.PillowWriter(fps=30))
    plt.show()

    # get response from user for saving costmap
    costmap_obj.save_to_disk()


def main():
    # PARAM SETUP
    # --- costmap --- #
    n = 300  # channel height
    m = 40  # channel width
    load_costmap_file = ""  # "sample_costmaps/random_obstacles_1.pk"

    # --- ship --- #
    start_pos = (20, 10, 0)  # (x, y, theta)
    goal_pos = (20, 282, 0)
    initial_heading = math.pi / 2
    turning_radius = 8
    vel = 10  # constant linear velocity of ship
    padding = 0  # padding around ship vertices to increase footprint when computing path costs

    # --- primitives --- #
    num_headings = 8

    # --- ice --- #
    num_obs = 130  # number of random ice obstacles
    min_r = 1  # min ice radius
    max_r = 8
    upper_offset = 20  # offset from top of costmap where ice stops
    lower_offset = 20  # offset from bottom of costmap where ice stops
    allow_overlap = False  # if True allow overlap in ice obstacles
    obstacle_density = 6
    obstacle_penalty = 0.25

    # --- A* --- #
    g_weight = 0.3  # cost = g_weight * g_score + h_weight * h_score
    h_weight = 0.7

    # --- pid --- #
    Kp = 3
    Ki = 0.08
    Kd = 0.5

    # -- misc --- #
    smooth_path = False  # if True run smoothing algorithm
    replan = False  # if True rerun A* search at each time step
    save_animation = False  # if True save animation and don't show it

    state_lattice_planner(g_weight=g_weight, h_weight=h_weight, costmap_file=load_costmap_file,
                          start_pos=start_pos, goal_pos=goal_pos, initial_heading=initial_heading, padding=padding,
                          turning_radius=turning_radius, vel=vel, num_headings=num_headings,
                          num_obs=num_obs, min_r=min_r, max_r=max_r, upper_offset=upper_offset,
                          lower_offset=lower_offset, allow_overlap=allow_overlap, obstacle_density=obstacle_density,
                          obstacle_penalty=obstacle_penalty, Kp=Kp, Ki=Ki, Kd=Kd,
                          save_animation=save_animation, smooth_path=smooth_path, replan=replan)


if __name__ == "__main__":
    main()
