import math
import time

import dubins
import numpy as np
import pymunk
from matplotlib import animation
from matplotlib import patches
from matplotlib import pyplot as plt
from pymunk.vec2d import Vec2d
from skimage import draw

from a_star_search import AStar
from cost_map import CostMap
from primitives import Primitives
from ship import Ship
from utils import heading_to_world_frame, plot_path


def generate_swath(ship: Ship, edge_set: np.ndarray, heading: int, prim: Primitives) -> dict:
    """
    Will have key of (edge, start heading)
    """
    swath_set = {}
    dist = lambda a, b: abs(a[0] - a[1])
    # why do we care about the max ship length from the centre of the ship?
    max_ship_length = np.ceil(max(dist(a, b) for a in ship.vertices for b in ship.vertices)).astype(int)
    start_pos = [prim.max_prim + max_ship_length] * 2 + [heading]

    for e in edge_set:
        e = tuple(e)
        array = np.zeros([(prim.max_prim + max_ship_length) * 2 + 1] * 2, dtype=bool)
        translated_e = np.asarray(e) + np.array([start_pos[0], start_pos[1], 0])

        theta_0 = heading_to_world_frame(start_pos[2], ship.initial_heading)
        theta_1 = heading_to_world_frame(translated_e[2], ship.initial_heading) % (2 * math.pi)
        dubins_path = dubins.shortest_path((start_pos[0], start_pos[1], theta_0),
                                           (translated_e[0], translated_e[1], theta_1),
                                           ship.turning_radius)

        configurations, _ = dubins_path.sample_many(0.5)

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

        swath_set[e, 0 + heading] = array
        swath_set[e, 2 + heading] = np.flip(array.T, 1)  # Rotate 90 degrees CCW
        swath_set[e, 4 + heading] = np.flip(np.flip(array, 1), 0)  # Rotate 180 degrees CCW
        swath_set[e, 6 + heading] = np.flip(array.T, 0)  # Rotate 270 degrees CCW

    return swath_set


def main():
    # Resolution is 10 m
    n = 600
    m = 70
    initial_heading = 2 * math.pi / 3
    turning_radius = 30  # 300 m turn radius
    ship_vertices = np.array([[-1, 5],
                              [1, 5],
                              [1, -5],
                              [-1, -5]])
    obstacle_penalty = 3
    start_pos = (35, 10, 0)  # (x, y, theta), possible values for theta 0 - 7 measured from ships positive x axis
    goal_pos = (35, 590, 0)

    # initialize costmap
    costmap_obj = CostMap(n, m, obstacle_penalty)

    # generate random obstacles
    costmap_obj.generate_obstacles(start_pos, goal_pos, num_obs=160, min_r=1, max_r=10,
                                   upper_offset=200, lower_offset=20, allow_overlap=False)

    # initialize ship object
    ship = Ship(ship_vertices, start_pos, goal_pos, initial_heading, turning_radius)

    # get the primitives
    prim = Primitives(scale=30, initial_heading=initial_heading)

    # generate swaths
    ordinal_swaths = generate_swath(ship, prim.edge_set_ordinal, 1, prim)
    cardinal_swaths = generate_swath(ship, prim.edge_set_cardinal, 0, prim)

    # initialize a star object
    a_star = AStar(g_weight=0.5, h_weight=0.5, cmap=costmap_obj,
                   primitives=prim, ship=ship)

    t0 = time.clock()
    worked, orig_cost, smoothed_edge_path, nodes_visited, x1, y1, x2, y2, orig_path = \
        a_star.search(start_pos, goal_pos, cardinal_swaths, ordinal_swaths)

    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    print("Hz", 1 / t1)
    print("smoothed path", smoothed_edge_path)

    smoothed_cost = costmap_obj.compute_path_cost(path=smoothed_edge_path.copy(), reverse_path=True,
                                                  turning_radius=turning_radius, ship_vertices=ship_vertices)
    # FIXME: this should be the same as `original_cost` !!
    recomputed_original_cost = costmap_obj.compute_path_cost(path=orig_path, reverse_path=False,
                                                             turning_radius=turning_radius, ship_vertices=ship_vertices)
    print("\nPath cost:\n\toriginal: {:.4f}\n\twith smoothing: {:.4f}\n".format(orig_cost, smoothed_cost))

    fig1, ax1 = plt.subplots(1, 2, figsize=(5, 10))

    # '''
    # FIXME: why regenerate again, can't we just do this in the smoothing step????
    if worked:
        ax1[0].imshow(costmap_obj.cost_map, origin='lower')
        PATH = smoothed_edge_path[::-1]
        path = np.zeros((2, 1))  # what is this used for?

        for i in range(np.shape(PATH)[0] - 1):
            P1 = PATH[i]
            P2 = PATH[i + 1]
            theta_0 = heading_to_world_frame(P1[2], initial_heading) % (2 * math.pi)
            theta_1 = heading_to_world_frame(P2[2], initial_heading) % (2 * math.pi)
            dubins_path = dubins.shortest_path((P1[0], P1[1], theta_0),
                                               (P2[0], P2[1], theta_1),
                                               turning_radius - 1e-4)
            configurations, _ = dubins_path.sample_many(0.2)
            # 0.01
            x = list()
            y = list()
            for config in configurations:
                x.append(config[0])
                y.append(config[1])
            ax1[0].plot(x, y, 'g')
            path = np.append(path, np.array([np.asarray(x).T, np.asarray(y).T]), axis=1)

        path = np.delete(path, 0, 1)
        print(np.shape(path))

        for obs in costmap_obj.grouped_obstacles:
            ax1[0].add_patch(patches.Polygon(obs['vertices'], True, fill=False))
        ax1[0].plot(x1, y1, 'bx')
        ax1[0].plot(x2, y2, 'gx')
    else:
        path = 0

    node_plot = np.zeros((n, m))
    print("nodes visited")
    for node in nodes_visited:
        # print(node)
        node_plot[node[1], node[0]] = node_plot[node[1], node[0]] + 1

    ax1[1].imshow(node_plot, origin='lower')
    # '''
    # '''
    print("Num of nodes expanded", np.sum(node_plot))
    plt.show()

    fig = plt.figure(figsize=(5, 10))
    plt.imshow(costmap_obj.cost_map)
    plt.show()
    # FIXME: skipping pymunk stuff for now
    exit()

    space = pymunk.Space()
    space.gravity = (0, 0)

    circles = []
    patch_list = []

    ship = Ship(space, initial_vel, start_pos[0], start_pos[1], start_pos[2])
    i = 0
    vs = np.zeros((5, 2))
    for ship_vertices in ship.shape.get_vertices():
        x, y = ship_vertices.rotated(ship.body.angle) + ship.body.position
        vs[i][0] = x
        vs[i][1] = y
        i += 1

    ship_patch = patches.Polygon(vs, True)
    # ship_patch = patches.Circle((ship.body.position.x, ship.body.position.y), 0.5)

    # TODO: update pymunk stuff
    for obs in costmap_obj.obstacles:
        circles.append(create_circle(space, *obs['centre'], obs['radius']))
        patch_list.append(patches.Circle(*obs['centre'], obs['radius'], fill=False))

    path = path.T
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
            heading = (math.atan2(velocity[1], velocity[0]) - math.pi / 2 + 2 * math.pi) % (2 * math.pi)
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

    fig2 = plt.figure()
    ax2 = plt.axes(xlim=(0, m), ylim=(0, n))
    ax2.set_aspect("equal")

    def init():
        ax2.add_patch(ship_patch)
        for circle, patch in zip(circles, patch_list):
            ax2.add_patch(patch)
        return []

    def animate(dt, ship_patch, ship, circles, patch_list):
        # print(dt)
        # 20 ms step size
        for x in range(10):
            space.step(2 / 100 / 10)

        ship_pos = (ship.body.position.x, ship.body.position.y)
        # print("path_node:", ship.path_pos, sep=" ")
        # print("ship pos:", ship_pos, sep=" ")
        # print("path pos:", path[ship.path_pos, :], sep=" ")
        # determine which part of the path ship is on and get translational/angular velocity for ship
        if ship.path_pos < np.shape(vel_path)[0]:
            ship.body.velocity = Vec2d(vel_path[ship.path_pos, 0], vel_path[ship.path_pos, 1])
            ship.body.angular_velocity = angular_vel[ship.path_pos]
            if dist(ship_pos, path[ship.path_pos, :]) < 0.01:
                ship.set_path_pos(ship.path_pos + 1)

        animate_ship(dt, ship_patch, ship)
        for circle, patch in zip(circles, patch_list):
            animate_obstacle(dt, circle, patch)
        return []

    def animate_ship(dt, patch, ship):
        heading = ship.body.angle
        R = np.asarray([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
        vs = np.asarray(ship.shape.get_vertices()) @ R + np.asarray(ship.body.position)
        patch.set_xy(vs)
        # pos_x = ship.body.position.x
        # pos_y = ship.body.position.y
        # patch.center = (pos_x, pos_y)
        return patch,

    def animate_obstacle(dt, circle, patch):
        pos_x = circle.body.position.x
        pos_y = circle.body.position.y
        patch.center = (pos_x, pos_y)
        return patch_list

    frames = np.shape(path)[0]
    anim = animation.FuncAnimation(fig2,
                                   animate,
                                   init_func=init,
                                   frames=frames,
                                   fargs=(ship_patch, ship, circles, patch_list,),
                                   interval=20,
                                   blit=True,
                                   repeat=False)

    plt.show()


if __name__ == "__main__":
    main()
