import math
import os
import pickle
import time
import copy

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
from utils import heading_to_world_frame


def generate_swath(ship: Ship, edge_set: np.ndarray, heading: int, prim: Primitives) -> dict:
    """
    Will have key of (edge, start heading)
    """
    swath_set = {}
    start_pos = [prim.max_prim + ship.max_ship_length // 2] * 2 + [heading]
    print(start_pos)

    for e in edge_set:
        e = tuple(e)
        array = np.zeros([(prim.max_prim + ship.max_ship_length // 2) * 2 + 1] * 2, dtype=bool)
        translated_e = np.asarray(e) + np.array([start_pos[0], start_pos[1], 0])

        theta_0 = heading_to_world_frame(start_pos[2], ship.initial_heading)
        theta_1 = heading_to_world_frame(translated_e[2], ship.initial_heading)
        dubins_path = dubins.shortest_path((start_pos[0], start_pos[1], theta_0),
                                           (translated_e[0], translated_e[1], theta_1),
                                           ship.turning_radius)

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

        # TODO: remove hardcode
        swath_set[e, 0 + heading] = array
        swath_set[e, 2 + heading] = np.flip(array.T, 1)  # Rotate 90 degrees CCW
        swath_set[e, 4 + heading] = np.flip(np.flip(array, 1), 0)  # Rotate 180 degrees CCW
        swath_set[e, 6 + heading] = np.flip(array.T, 0)  # Rotate 270 degrees CCW

    return swath_set


def snap_to_lattice(start_pos, goal_pos, initial_heading, turning_radius):
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
    abs_init_heading = heading_to_world_frame(start_pos[2], initial_heading)
    abs_goal_heading = heading_to_world_frame(goal_pos[2], initial_heading)
    diff = abs_goal_heading - abs_init_heading

    if diff < 0:
        diff = diff + (2 * math.pi)

    # check if x,y coordinates or heading are off lattice
    if diff_y != 0 or diff_x != 0 or diff % (math.pi / 4) != 0:
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
        new_theta = round(diff / (math.pi / 4))
        if new_theta > 7:
            new_theta = new_theta - 8

        # rotate coordinates back to original frame
        new_goal = np.array([[new_goal_x], [new_goal_y]])
        new_goal = R.T @ new_goal
        goal_pos = (
            round(new_goal[0][0] + start_pos[0], 5),
            round(new_goal[1][0] + start_pos[1], 5),
            new_theta
        )

    return goal_pos


def create_polygon(space, staticBody, vertices, x, y, density, radius):
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Poly(body, vertices, radius=radius)
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

def generate_path_traj(path):
    heading_list = np.zeros(np.shape(path)[0])
    vel_path = np.zeros((np.shape(path)[0] - 1, np.shape(path)[1]))
    angular_vel = np.zeros(np.shape(vel_path)[0])

    for i in range(np.shape(vel_path)[0]):
        point1 = path[i, :]
        point2 = path[i + 1, :]
        velocity = point2 - point1
        if velocity[0] == 0 or velocity[1] == 0:
            if velocity[0] > 0:
                heading = math.pi / 2
            elif velocity[0] < 0:
                heading = 3 * math.pi / 2
            elif velocity[1] > 0:
                heading = 0
            else:
                heading = math.pi
        else:
            heading = (math.atan2(velocity[1], velocity[0]) - math.pi / 2 + 2 * math.pi) % (2 * math.pi)  # FIXME: update
        heading_list[i] = heading
        # print("velocity: ", velocity, sep=" ")
        vel_path[i, :] = velocity.T * 50
        # print(tuple(vel_path[i,:]))

    # set initial heading and final heading
    heading_list[0] = 0
    heading_list[-1] = 0

    # Estimate angular velocity at each point from current and next heading
    for i in range(np.shape(angular_vel)[0]):
        raw = heading_list[i + 1] - heading_list[i]
        turn = min((-abs(raw)) % (2 * math.pi), abs(raw) % (2 * math.pi))
        if raw == 0:
            direction = -1
        else:
            direction = -abs(raw) / raw
        angular_vel[i] = direction * turn * 30

    return(vel_path, angular_vel)


# FIXME: improve
def plot_path(fig1, costmap_obj, smoothed_edge_path, initial_heading, turning_radius, smooth_path, prim, x1, x2, y1, y2, node_plot, nodes_visited):
    plt.close(fig1)
    fig1, ax1 = plt.subplots(1, 2, figsize=(5, 10))
    ax1[0].imshow(costmap_obj.cost_map, origin='lower')
    PATH = smoothed_edge_path[::-1]
    path = np.zeros((2, 1))  # what is this used for?

    for i in range(np.shape(PATH)[0] - 1):
        P1 = PATH[i]
        P2 = PATH[i + 1]
        theta_0 = heading_to_world_frame(P1[2], initial_heading)
        theta_1 = heading_to_world_frame(P2[2], initial_heading)
        dubins_path = dubins.shortest_path((P1[0], P1[1], theta_0),
                                           (P2[0], P2[1], theta_1),
                                           turning_radius - 1e-4)
        configurations, _ = dubins_path.sample_many(0.2)
        # 0.01
        x = []
        y = []
        for config in configurations:
            x.append(config[0])
            y.append(config[1])

        if not smooth_path and False:  # only want to show primitives on un smoothed path
            if (P1[2] * 45) % 90 == 0:
                edge_set = prim.edge_set_cardinal
            else:
                edge_set = prim.edge_set_ordinal
            for e in edge_set:
                p2 = AStar.concat(P1, e)
                theta_1 = heading_to_world_frame(p2[2], initial_heading) % (2 * math.pi)
                dubins_path = dubins.shortest_path((P1[0], P1[1], theta_0), (p2[0], p2[1], theta_1), turning_radius - 0.001)
                configurations, _ = dubins_path.sample_many(0.2)
                x3 = []
                y3 = []
                for config in configurations:
                    x3.append(config[0])
                    y3.append(config[1])
                ax1[0].plot(x3, y3, 'r')

        ax1[0].plot(x, y, 'g')
        path = np.append(path, np.array([np.asarray(x).T, np.asarray(y).T]), axis=1)

    path = np.delete(path, 0, 1)

    for obs in costmap_obj.obstacles:
        ax1[0].add_patch(patches.Polygon(obs['vertices'], True, fill=False))
    ax1[0].plot(x1, y1, 'bx')
    ax1[0].plot(x2, y2, 'gx')

    for node in nodes_visited:
        r, c = int(round(node[1])), int(round(node[0]))
        node_plot[r, c] = node_plot[r, c] + 1

    ax1[1].imshow(node_plot, origin='lower')
    return fig1, path, node_plot, ax1


def main():
    load_costmap_file = ""  # "sample_costmaps/random_obstacles_1.pk"
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
    start_pos = (20, 10, 0)  # (x, y, theta), possible values for theta 0 - 7 measured from ships positive x axis
    goal_pos = (20, 282, 0)
    print("GOAL", goal_pos)
    smooth_path = False

    # load costmap object from file if specified
    if load_costmap_file:
        with open(load_costmap_file, "rb") as fd:
            costmap_obj = pickle.load(fd)
    else:
        # initialize costmap
        costmap_obj = CostMap(n, m, obstacle_penalty)

        # generate random obstacles
        costmap_obj.generate_obstacles(start_pos, goal_pos, num_obs=130, min_r=1, max_r=8,
                                       upper_offset=20, lower_offset=20, allow_overlap=False)

    # initialize ship object
    ship = Ship(ship_vertices, start_pos, initial_heading, turning_radius)
    print("TURN RADIUS", ship.calc_turn_radius(45, 2))
    # get the primitives
    prim = Primitives(scale=turning_radius, initial_heading=initial_heading)

    # generate swaths
    ordinal_swaths = generate_swath(ship, prim.edge_set_ordinal, 1, prim)
    cardinal_swaths = generate_swath(ship, prim.edge_set_cardinal, 0, prim)

    # test_image = transform.rotate(ordinal_swaths[tuple(prim.edge_set_ordinal[0]), 1], -10)
    # plt.imshow(test_image, origin='lower')
    # plt.show()

    # initialize a star object
    a_star = AStar(g_weight=0.5, h_weight=0.5, cmap=costmap_obj,
                   primitives=prim, ship=ship, first_initial_heading=initial_heading)

    t0 = time.clock()
    worked, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
        a_star.search(start_pos, goal_pos, cardinal_swaths, ordinal_swaths, smooth_path)

    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    print("Hz", 1 / t1)
    print("smoothed path", smoothed_edge_path)

    recomputed_original_cost, og_length = costmap_obj.compute_path_cost(path=orig_path, ship=ship, reverse_path=True)
    smoothed_cost, smooth_length = costmap_obj.compute_path_cost(path=smoothed_edge_path.copy(), ship=ship, reverse_path=True)
    straight_path_cost, straight_length = costmap_obj.compute_path_cost(path=[start_pos, goal_pos], ship=ship)
    print("\nPath cost:"
          "\n\toriginal path:  {:.4f}"
          "\n\twith smoothing: {:.4f}"
          "\n\tstraight path:  {:.4f}\n".format(recomputed_original_cost, smoothed_cost, straight_path_cost))
    print("\nPath length:"
          "\n\toriginal path:  {:.4f}"
          "\n\twith smoothing: {:.4f}"
          "\n\tstraight path:  {:.4f}\n".format(og_length, smooth_length, straight_length))
    try:
        assert smoothed_cost <= recomputed_original_cost <= straight_path_cost,\
            "smoothed cost should be less than original cost and original cost should be less than straight cost"
    except AssertionError as error:
        print(error)
        costmap_obj.save_to_disk()

    fig1, ax1 = plt.subplots(1, 2, figsize=(5, 10))

    # '''
    # FIXME: why regenerate again, can't we just do this in the smoothing step????
    node_plot = np.zeros((n, m))
    if worked:
        fig1, path, node_plot, _ = plot_path(fig1, costmap_obj, smoothed_edge_path, initial_heading, turning_radius, smooth_path, prim, x1, x2, y1, y2, node_plot, nodes_visited)
    else:
        path = 0

    '''
    node_plot = np.zeros((n, m))
    for node in nodes_visited:
        r, c = int(round(node[1])), int(round(node[0]))
        node_plot[r, c] = node_plot[r, c] + 1
            ax1[1].imshow(node_plot, origin='lower')
    print("Num of nodes expanded", np.sum(node_plot))
    '''

    space = pymunk.Space()
    space.add(ship.body, ship.shape)
    space.gravity = (0, 0)
    staticBody = space.static_body  # create a static body for friction constraints

    polygons = []
    patch_list = []

    print("HEADING", ship.body.angle)
    i = 0
    vs = np.zeros((5, 2))
    for ship_vertices in ship.shape.get_vertices():
        x, y = ship_vertices.rotated(ship.body.angle) + ship.body.position
        vs[i][0] = x
        vs[i][1] = y
        i += 1

    ship_patch = patches.Polygon(vs, True, color='green')

    # TODO: update pymunk stuff
    print("GENERATE OBSTACLES")
    for obs in costmap_obj.obstacles:
        polygons.append(
            create_polygon(space, staticBody, (obs['vertices'] - np.array(obs['centre'])).tolist(),
                           *obs['centre'], density, obs['radius'])
        )
        patch_list.append(patches.Polygon(obs['vertices'], True))

    path = path.T
    vel_path, angular_vel = generate_path_traj(path)


    fig2 = plt.figure()
    ax2 = plt.axes(xlim=(0, m), ylim=(0, n))
    ax2.set_aspect("equal")

    def init():
        ax2.add_patch(ship_patch)
        for patch in patch_list:
            ax2.add_patch(patch)
        return []

    def animate(dt, ship_patch, ship, polygons, patch_list, vel_list, ang_vel_list, path, fig1, ordinal_swaths, cardinal_swaths):
        # print(dt)
        # 20 ms step size
        for x in range(10):
            space.step(2 / 100 / 10)

        ship_pos = (ship.body.position.x, ship.body.position.y)

        if (dt % 50  == 0 and dt != 0):
            print("\nNEXT STEP")
            curr_pos = (ship_pos[0], ship_pos[1], ship.body.angle)
            snapped_goal = snap_to_lattice(curr_pos, goal_pos, ship.initial_heading, turning_radius)  # FIXME
            curr_pos = (ship_pos[0], ship_pos[1], 0)  # straight ahead of boat is 0

            ship.initial_heading = ship.body.angle + a_star.first_initial_heading
            prim.rotate(ship.body.angle, orig=True)

            ordinal_swaths, cardinal_swaths = prim.update_swath(theta=ship.body.angle,
                                                                ord_swath=ordinal_swaths,
                                                                card_swath=cardinal_swaths)

            print("INITIAL HEADING", ship.initial_heading)
            print("ANGLE", ship.body.angle)
            print("NEW GOAL", snapped_goal)
            print("NEW START", curr_pos)
            t0 = time.clock()
            worked, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
                a_star.search(curr_pos, snapped_goal, cardinal_swaths, ordinal_swaths, smooth_path)
            t1 = time.clock() - t0
            print("PLAN TIME", t1)
            if worked:
                print("Replanned Path", smoothed_edge_path)
                '''
                path = plot_path(ax1, costmap_obj, smoothed_edge_path, initial_heading, turning_radius, smooth_path, prim, x1, x2, y1, y2)
                path = path.T
                vel_list, ang_vel_list = generate_path_traj(path)
                print(np.shape(vel_path))
                ship.set_path_pos(0)               
                '''
                costmap_obj.update(polygons)
                node_plot = np.zeros((n, m))
                fig1, _, node_plot, ax1 = plot_path(fig1, costmap_obj, smoothed_edge_path, ship.initial_heading, turning_radius, smooth_path, prim, x1, x2, y1, y2, node_plot, nodes_visited)
                plt.show(block=False)
                # print("got out")
        # '''

        # determine which part of the path ship is on and get translational/angular velocity for ship
        if ship.path_pos < np.shape(vel_list)[0]:
            ship.body.velocity = Vec2d(vel_list[ship.path_pos, 0], vel_list[ship.path_pos, 1])
            ship.body.angular_velocity = ang_vel_list[ship.path_pos]
            if a_star.dist(ship_pos, path[ship.path_pos, :]) < 0.01:
                ship.set_path_pos(ship.path_pos + 1)

        animate_ship(dt, ship, ship_patch)
        for poly, patch in zip(polygons, patch_list):
            animate_obstacle(dt, poly, patch)
        return []

    def animate_ship(dt, ship, patch):
        heading = ship.body.angle
        R = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        vs = np.asarray(ship.shape.get_vertices()) @ R + np.asarray(ship.body.position)
        patch.set_xy(vs)
        return patch,

    def animate_obstacle(dt, polygon, patch):
        heading = polygon.body.angle
        R = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        vs = np.asarray(polygon.get_vertices()) @ R + np.asarray(polygon.body.position)
        patch.set_xy(vs)
        return patch_list

    print("START ANIMATION")
    frames = np.shape(path)[0]
    anim = animation.FuncAnimation(fig2,
                                   animate,
                                   init_func=init,
                                   frames=frames,
                                   fargs=(ship_patch, ship, polygons, patch_list, vel_path, angular_vel, path, fig1, ordinal_swaths, cardinal_swaths, ),
                                   interval=20,
                                   blit=True,
                                   repeat=False)

    plt.show()

    # get response from user for saving costmap
    costmap_obj.save_to_disk()


if __name__ == "__main__":
    main()
