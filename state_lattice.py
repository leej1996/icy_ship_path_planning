import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
import cost_map as cm
import pymunk
from pymunk.vec2d import Vec2d
from queue import PriorityQueue
from skimage.draw import polygon
import copy
import dubins
import time
import random

from primitives import Primitives



class CustomPriorityQueue(PriorityQueue):
    def _put(self, item):
        return super()._put((self._get_priority(item), item))  # prioritized based on f score

    def _get(self):
        return super()._get()[1]

    def _get_priority(self, item):
        return item[1]

    def _update(self, item, update_value):
        self.queue.remove(((item[1]), (item[0], item[1])))  # custom queue to update the priorities of objects
        self._put((item[0], update_value))


def generate_obstacles(start, goal, n, m, num_obs, min_r, max_r, upper_offset, lower_offset):
    obstacles = list()
    protection = 0
    #print("max y", goal[1] - offset - max_r)
    x = random.randint(max_r, m - max_r - 1)
    y = random.randint(start[1] + lower_offset + max_r, goal[1] - upper_offset - max_r)
    r = random.randint(min_r, max_r)
    obstacles.append([y, x, r])
    while len(obstacles) < num_obs - 1:
        near_obs = False
        x = random.randint(max_r, m - max_r - 1)
        y = random.randint(start[1] + lower_offset + max_r, goal[1] - upper_offset - max_r)
        r = random.randint(min_r, max_r)
        # check if obstacles overlap
        for obs in obstacles:
            if ((x - obs[1]) ** 2 + (y - obs[0]) ** 2) ** 0.5 < obs[2] + r:
                near_obs = True
                continue
        if not near_obs:
            obstacles.append([y, x, r])

        protection += 1

        if protection > 300:
            break

    return obstacles


def generate_swath(vertices, edge_set, turning_radius, heading):
    '''
    Will have key of (edge, start heading)
    '''
    swath_set = dict()
    start_pos = (155, 155, heading)

    for e in edge_set:
        e = tuple(e)
        #array = np.zeros((11, 11), dtype=bool)
        array = np.zeros((311,311), dtype=bool) # (max size (150) + max length of boat rounded up (5)) * 2 + 1
        translated_e = np.asarray(e) + np.array([start_pos[0], start_pos[1], 0])
        dubins_path = dubins.shortest_path((start_pos[0], start_pos[1], math.radians((start_pos[2] + 2) * 45)),
                                           (translated_e[0], translated_e[1], math.radians((translated_e[2] + 2) * 45) % (2 * math.pi)),
                                           turning_radius)
        configurations, _ = dubins_path.sample_many(0.5)

        for config in configurations:
            x_cell = int(round(config[0]))
            y_cell = int(round(config[1]))
            theta = config[2] - math.pi/2
            R = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rot_vi = np.round(np.array([[x_cell], [y_cell]]) + R @ vertices.T).astype(int)

            rr, cc = polygon(rot_vi[1, :], rot_vi[0, :])
            array[rr, cc] = True

        swath_set[e, 0 + heading] = array
        swath_set[e, 2 + heading] = np.flip(array.T, 1)  # Rotate 90 degrees CCW
        swath_set[e, 4 + heading] = np.flip(np.flip(array, 1), 0)  # Rotate 180 degrees CCW
        swath_set[e, 6 + heading] = np.flip(array.T, 0)  # Rotate 270 degrees CCW

    return swath_set


def get_swath(e, n, b, start_pos, swath_set):
    swath = np.zeros((n, b), dtype=bool)
    heading = int(start_pos[2])
    swath1 = swath_set[tuple(e), heading]

    # swath mask has starting node at the centre (11x11) and want to put at the starting node of currently expanded node
    # in the costmap, need to remove the extra columns/rows of the swath mask

    swath_size = swath1.shape[0]
    min_y = start_pos[1] - 155
    max_y = start_pos[1] + 156
    min_x = start_pos[0] - 155
    max_x = start_pos[0] + 156
    # Too far to the right
    if max_x >= b:
        overhang = max_x - (b - 1)
        swath1 = np.delete(swath1, slice(swath_size - overhang, swath_size), axis=1)
        max_x = b - 1
    # Too far to the left
    if min_x < 0:
        overhang = abs(min_x)
        swath1 = np.delete(swath1, slice(0, overhang), axis=1)
        min_x = 0
    # Too close to the top
    if max_y >= n:
        overhang = max_y - (n - 1)
        swath1 = np.delete(swath1, slice(swath_size - overhang, swath_size), axis=0)
        max_y = n - 1
    # Too close to the bottom
    if min_y < 0:
        overhang = abs(min_y)
        swath1 = np.delete(swath1, slice(0, overhang), axis=0)
        min_y = 0
    swath[min_y:max_y, min_x:max_x] = swath1

    return swath


def Concat(x, y):
    """
    given two points x,y in the lattice, find the concatenation x + y
    """
    rot = x[2] * math.pi / 4  # starting heading
    p1 = [x[0], x[1]]
    p2_theta = y[2] * math.pi / 4  # edge heading
    p2 = [y[0], y[1]]

    if x[2] % 2 == 0:
        # cardinal
        heading = p2_theta + rot
    else:
        # ordinal
        rot = rot - math.pi / 4
        heading = p2_theta + rot

    R = np.array([[math.cos(rot), -math.sin(rot)],
                  [math.sin(rot), math.cos(rot)]])
    multiplication = np.matmul(R, np.transpose(np.asarray(p2)))

    result = np.asarray(p1) + multiplication

    while heading >= 2 * math.pi:
        heading = heading - 2 * math.pi
    heading = heading / (math.pi / 4)

    return (int(result[0]), int(result[1]), int(round(heading)))


def is_point_in_set(point, set):
    for point1 in set:
        if dist(point, point1) < 0.001 and abs(point[2] - point1[2]) < 0.001:
            return True, point1
    return False, False


def dist(a, b):
    # Euclidean distance
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def heuristic(p_initial, p_final, turning_radius):
    """
    The Dubins' distance from initial to final points.
    """
    p1 = (p_initial[0], p_initial[1], math.radians((p_initial[2] * 45 + 90) % 360))
    p2 = (p_final[0], p_final[1], math.radians((p_final[2] * 45 + 90) % 360))
    path = dubins.shortest_path(p1, p2, turning_radius)
    return path.path_length()


def near_obstacle(node, list_of_obstacles):
    for obs in list_of_obstacles:
        # check if ship is within radius + 5 squares of the center of obstacle, then do swath
        if dist(node, (obs[1], obs[0])) < obs[2] + 5 and not past_obstacle(node, obs):
            return True
    return False


def past_obstacle(node, obs):
    # also check if ship is past all obstacles (under assumption that goal is always positive y direction from start)
    # obstacle y coord + radius
    if node[1] > obs[0] + obs[2]:
        return True
    else:
        return False


def path_smoothing(path, path_length, cost_map, turning_radius, start, goal, nodes, n, m):
    print("Attempt Smoothing")
    total_length = np.sum(path_length)
    probabilities = np.asarray(path_length) / total_length
    # print(probabilities)
    x = list()
    y = list()

    for vi in path:
        x.append(vi[0])
        y.append(vi[1])

    segments = np.sort(np.random.choice(np.arange(len(path)), nodes, p=probabilities))

    added_x = list()
    added_y = list()
    counter = 0
    offset = 0
    while counter < len(segments):
        node_id = segments[counter]
        num_values = np.shape(segments[segments == node_id])[0]
        node = path[node_id + offset]
        prev_node = path[node_id + offset - 1]
        prim = dubins.shortest_path((prev_node[0], prev_node[1], math.radians((prev_node[2] + 2) * 45)),
                                    (node[0], node[1], math.radians((node[2] + 2) * 45)), turning_radius)
        configurations, _ = prim.sample_many(0.1)
        values = [configurations[int(i)] for i in
                  np.linspace(0, len(configurations), num=num_values + 2, endpoint=False)]
        values.pop(0)
        values.pop()

        inc = 1
        for v in values:
            heading = v[2] - math.pi / 2
            if 0 <= v[0] < m and 0 <= v[1] < n:
                added_x.append(v[0])
                added_y.append(v[1])
                if heading < 0:
                    heading = heading + 2 * math.pi

                path.insert(node_id - 1 + inc + counter, (v[0], v[1], heading / (math.pi / 4)))
                inc += 1
        counter = counter + num_values
        offset = offset + inc - 1

    print(len(path))
    prev = dict()
    smooth_cost = dict()
    remove_nodes = list()
    for vi in path:
        #if 0 <= vi[0] < m and 0 <= vi[1] < n:
        smooth_cost[vi] = math.inf
        prev[vi] = None
        #else:
            #remove_nodes.append(vi)
    smooth_cost[path[0]] = 0

    #for node in remove_nodes:
        #path.remove(node)

    i = 0
    for vi in path:
        for vj in path[i + 1:]:
            # determine cost between node vi and vj
            swath = [[int(round(vi[0])), int(round(vi[1]))]]
            swath_cost = 0
            dubins_path = dubins.shortest_path((vi[0], vi[1], math.radians((vi[2] + 2) * 45) % (2 * math.pi)),
                                               (vj[0], vj[1], math.radians((vj[2] + 2) * 45) % (2 * math.pi)),
                                               turning_radius)
            configurations, _ = dubins_path.sample_many(0.4)
            for config in configurations:
                x_cell = int(round(config[0]))
                y_cell = int(round(config[1]))
                if 0 <= x_cell < m and 0 <= y_cell < n:
                    if [x_cell, y_cell] not in swath:
                        # if x_cell not in swath1 and y_cell not in swath2:
                        swath.append([x_cell, y_cell])
                        swath_cost += cost_map[y_cell, x_cell]

            adj_cost = swath_cost + dubins_path.path_length()
            if smooth_cost[vi] + adj_cost < smooth_cost[vj]:
                smooth_cost[vj] = smooth_cost[vi] + adj_cost
                prev[vj] = vi

        i += 1

    smooth_path = list()
    smooth_path.append(goal)
    node = goal
    while node != start:
        node = prev[node]
        smooth_path.append(node)
    return smooth_path, x, y, added_x, added_y


def a_star(start, goal, turning_radius, n, m, cost_map, card_edge_set, ord_edge_set, cardinal_swath, ordinal_swath,
           list_of_obstacles):
    # theta is measured ccw from y axis
    a = 0.2
    b = 0.8
    free_path_interval = 2
    generation = 0  # number of nodes expanded
    openSet = dict()  # set of nodes considered for expansion
    print("start", start)
    openSet[start] = generation
    closedSet = []
    cameFrom = dict()
    cameFrom[start] = None
    cameFrom_by_edge = dict()
    cameFrom_by_edge[start] = None
    # cost from start
    g_score = dict()
    g_score[start] = 0
    # f score (g score + heuristic) (estimation of cost to goal)
    f_score = dict()
    f_score[start] = heuristic(start, goal, turning_radius)
    # path length between nodes
    path_length = dict()
    path_length[start] = 0
    # heading_delta = dict()
    # heading_delta[start] = 0
    # priority queue of all visited node f scores
    f_score_open_sorted = CustomPriorityQueue()
    f_score_open_sorted.put((start, f_score[start]))  # put item in priority queue

    # while np.shape(openSet)[0] != 0:
    while len(openSet) != 0:
        # node[0] = x position
        # node[1] = y position
        # node[2] = theta (heading)

        node = f_score_open_sorted.get()[0]

        # print("Generation: ", generation, sep=" ")
        #print("NODE:", node, sep=" ")

        # If ship past all obstacles, calc direct dubins path to goal

        if generation % free_path_interval == 0 and node != goal:
            past_all_obs = True
            for obs in list_of_obstacles:
                if not past_obstacle(node, obs):
                    past_all_obs = False
                    break

            if past_all_obs:
                print("Found path to goal")
                cameFrom[goal] = node
                path_length[goal] = heuristic(node, goal, turning_radius)
                f_score[goal] = g_score[node] + path_length[goal]
                pred = goal
                node = pred



        if node == goal:
            print("Found path")
            path = list()
            new_path_length = list()
            print("goal", goal)
            path_heading_delta = list()
            cameFrom[goal] = cameFrom[node]
            path.append(goal)
            new_path_length.append(path_length[goal])

            while node != start:
                pred = cameFrom[node]
                node = pred
                path.append(node)
                new_path_length.append(path_length[node])
                # if heading_delta[node] > 2:
                # heading_delta[node] = abs(heading_delta[node] - 8)
                # path_heading_delta.append(heading_delta[node])

            # print(path_heading_delta)
            # print(path)

            path.reverse()  # path: start -> goal
            new_path_length.reverse()
            print("path", path)
            add_nodes = int(len(path))  # number of nodes to add in the path smoothing algorithm

            t0 = time.clock()
            smooth_path, x1, y1, x2, y2 = path_smoothing(path, new_path_length, cost_map, turning_radius, start, goal,
                                                         add_nodes, n, m)
            t1 = time.clock() - t0
            print("smooth time", t1)
            return (True, f_score[goal], smooth_path, closedSet, x1, y1, x2, y2)


        openSet.pop(node)
        closedSet.append(node)

        if (node[2] * 45) % 90 == 0:
            edge_set = card_edge_set
            swath_set = cardinal_swath
        else:
            edge_set = ord_edge_set
            swath_set = ordinal_swath

        for e in edge_set:
            #print("edge:", e, sep=" ")
            neighbour = Concat(node, e)
            #print("neighbour:",neighbour, sep=" ")

            if 0 <= neighbour[0] < m and 0 <= neighbour[1] < n:
                # print("neighbour is valid")
                # check if point is in closed set
                neighbour_in_closed_set, closed_set_neighbour = is_point_in_set(neighbour, closedSet)
                if neighbour_in_closed_set:
                    continue

                # If near obstacle, check cost map to find cost of swath
                if near_obstacle(node, list_of_obstacles):
                    swath = get_swath(e, n, m, node, swath_set)
                    mask = cost_map[swath]
                    swath_cost = np.sum(mask)
                else:
                    swath_cost = 0

                temp_path_length = heuristic(node, neighbour, turning_radius)
                cost = swath_cost + temp_path_length
                temp_g_score = g_score[node] + cost

                if neighbour in openSet:
                    neighbour_in_open_set = True
                    open_set_neighbour = neighbour
                else:
                    neighbour_in_open_set = False
                    open_set_neighbour = False

                if not neighbour_in_open_set:
                    # print("new node")
                    heuristic_value = heuristic(neighbour, goal, turning_radius)
                    openSet[neighbour] = generation
                    cameFrom[neighbour] = node
                    cameFrom_by_edge[neighbour] = e
                    path_length[neighbour] = temp_path_length
                    # heading_delta[neighbour] = abs(neighbour[2] - node[2])
                    g_score[neighbour] = temp_g_score
                    f_score[neighbour] = a * g_score[neighbour] + b * heuristic_value
                    f_score_open_sorted.put((neighbour, f_score[neighbour]))
                elif neighbour_in_open_set and temp_g_score < g_score[open_set_neighbour]:
                    open_set_neighbour_heuristic_value = heuristic(open_set_neighbour, goal, turning_radius)
                    # print("found cheaper cost to node")
                    # print(open_set_neighbour)
                    cameFrom[open_set_neighbour] = node
                    cameFrom_by_edge[open_set_neighbour] = e
                    path_length[open_set_neighbour] = temp_path_length
                    g_score[open_set_neighbour] = temp_g_score
                    f_score_open_sorted._update((open_set_neighbour, f_score[open_set_neighbour]),
                                                a * g_score[
                                                    open_set_neighbour] + b * open_set_neighbour_heuristic_value)
                    f_score[open_set_neighbour] = a * g_score[
                        open_set_neighbour] + b * open_set_neighbour_heuristic_value
        generation = generation + 1
    return (False, 'Fail', 'Fail', 'Fail')


def create_circle(space, x, y, r):
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Circle(body, r)
    shape.density = 3
    space.add(body, shape)
    return shape


def generate_polygon(n, N, x_pos, y_pos):
    # generate two lists of x and y of N random integers between 0 and n
    x = [random.uniform(0, n) for _ in range(N)]
    y = [random.uniform(0, n) for _ in range(N)]

    # sort both lists
    x.sort()
    y.sort()

    x_max = x[-1]
    y_max = y[-1]
    x_min = x[0]
    y_min = y[0]

    lastTop = x_min
    lastBot = x_min
    xVec = list()

    for i in range(1, N - 1):
        val = x[i]
        if bool(random.getrandbits(1)):
            xVec.append(val - lastTop)
            lastTop = val
        else:
            xVec.append(lastBot - val)
            lastBot = val

    xVec.append(x_max - lastTop)
    xVec.append(lastBot - x_max)

    lastLeft = y_min
    lastRight = y_min
    yVec = list()

    for i in range(1, N - 1):
        val = y[i]
        if bool(random.getrandbits(1)):
            yVec.append(val - lastLeft)
            lastLeft = val
        else:
            yVec.append(lastRight - val)
            lastRight = val

    yVec.append(y_max - lastLeft)
    yVec.append(lastRight - y_max)
    random.shuffle(yVec)

    pairs = zip(xVec, yVec)
    sorted_pairs = sorted(pairs, key=lambda pair: math.atan2(pair[0], pair[1]))

    minPolygonX = 0
    minPolygonY = 0
    x = 0
    y = 0
    points = list()

    for pair in sorted_pairs:
        points.append((x, y))
        x += pair[0]
        y += pair[1]
        minPolygonX = min(minPolygonX, x)
        minPolygonY = min(minPolygonY, y)

    x_shift = x_min - minPolygonX
    y_shift = y_min - minPolygonY

    points = np.asarray(points) + np.array([x_shift, y_shift]).T + np.array([x_pos, y_pos]).T
    return points


class Ship:
    def __init__(self, space, v, x, y, theta):
        self.vertices = [(0, 2), (0.5, 1), (0.5, -1), (-0.5, -1), (-0.5, 1)]
        self.body = pymunk.Body(1, 100, body_type=pymunk.Body.KINEMATIC)
        self.body.position = (x, y)
        self.body.velocity = v
        self.body.angle = math.radians(theta)
        self.shape = pymunk.Poly(self.body, self.vertices)
        # self.shape = pymunk.Circle(self.body, 0.5)
        space.add(self.body, self.shape)
        self.path_pos = 0

    def set_path_pos(self, path_pos):
        self.path_pos = path_pos


def calc_turn_radius(rate, speed):
    '''
    rate: deg/min
    speed: knots
    '''
    theta = rate * math.pi / 180  # convert to rads
    s = speed * 30.8667  # convert to m
    turn_radius = s/theta
    return turn_radius


def main():
    # resolution is 1 turning radius TBD
    # Resolution is 10 m
    # ice tank is 76 x 12 m
    #n = 76
    #m = 20
    n = 400
    m = 50
    theta = 0  # Possible values: 0 - 7, each number should be multiplied by 45 degrees (measured CCW from up)
    #turning_radius = 0.999
    turning_radius = 29.9999 # 300 m turn radius
    scale = 3
    start_pos = (40, 10, theta)
    goal_pos = (20, 390, 0)
    cost_map = np.zeros((n, m))

    # ship vertices
    v = np.array([[-1, 4],
                  [1, 4],
                  [1, -4],
                  [-1, -4]])
    #vx = np.array([-1, 1, 1, -1])
    #vy = np.array([4, 4, -4, -4])

    list_of_obstacles = generate_obstacles(start_pos, goal_pos, n, m, 80, 1, 10, 70, 20)
    for obs in list_of_obstacles:
        cost_map = cm.create_circle(obs, cost_map, scale)
    print(list_of_obstacles)
    # y is pointing up, x is pointing to the right
    # must rotate all swaths pi/4 CCW to be facing up
    prim = Primitives(scale=30, rotate=True)

    ordinal_swaths = generate_swath(v, prim.edge_set_ordinal, turning_radius, 1)
    cardinal_swaths = generate_swath(v, prim.edge_set_cardinal, turning_radius, 0)

    #turn_radius = calc_turn_radius(15, 4)
    #print("turn radius", turn_radius)
    t0 = time.clock()
    worked, L, edge_path, nodes_visited, x1, y1, x2, y2 = a_star(start_pos, goal_pos, turning_radius, n, m, cost_map,
                                                                 prim.edge_set_cardinal,
                                                                 prim.edge_set_ordinal, cardinal_swaths, ordinal_swaths,
                                                                 list_of_obstacles)
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    print("Hz", 1 / t1)

    fig1, ax1 = plt.subplots(1, 2)

    # '''
    if worked:
        ax1[0].imshow(cost_map, origin='lower')
        xmax = 0
        ymax = 0
        PATH = [i for i in edge_path[::-1]]
        path = np.zeros((2, 1))

        for i in range(np.shape(PATH)[0] - 1):
            P1 = PATH[i]
            P2 = PATH[i + 1]
            dubins_path = dubins.shortest_path((P1[0], P1[1], math.radians(P1[2] * 45 + 90) % (2 * math.pi)),
                                               (P2[0], P2[1], math.radians(P2[2] * 45 + 90) % (2 * math.pi)),
                                               turning_radius)
            configurations, _ = dubins_path.sample_many(0.2)
            # 0.01
            x = list()
            y = list()
            for config in configurations:
                x.append(config[0])
                y.append(config[1])
                if config[0] > xmax:
                    xmax = config[0]
                if config[1] > ymax:
                    ymax = config[1]
            ax1[0].plot(x, y, 'g')
            path = np.append(path, np.array([np.asarray(x).T, np.asarray(y).T]), axis=1)

        path = np.delete(path, 0, 1)
        print(np.shape(path))

        for obs in list_of_obstacles:
            ax1[0].add_patch(patches.Circle((obs[1], obs[0]), obs[2], fill=False))
            vs = generate_polygon(obs[2] * 2, 6, obs[1] - obs[2], obs[0] - obs[2])
            poly = patches.Polygon(vs, True)
            ax1[0].add_patch(poly)
        ax1[0].plot(x1, y1, 'bx')
        ax1[0].plot(x2, y2, 'gx')
    else:
        path = 0

    node_plot = np.zeros((n, m))
    print("nodes visited")
    for node in nodes_visited:
        #print(node)
        node_plot[node[1], node[0]] = node_plot[node[1], node[0]] + 1

    ax1[1].imshow(node_plot, origin='lower')
    # '''
    # '''
    print("Total Cost:", L, sep=" ")
    print("Num of nodes expanded", np.sum(node_plot))

    space = pymunk.Space()
    space.gravity = (0, 0)
    initial_vel = Vec2d(0, 0)

    circles = []
    patch_list = []

    ship = Ship(space, initial_vel, start_pos[0], start_pos[1], start_pos[2])
    i = 0
    vs = np.zeros((5, 2))
    for v in ship.shape.get_vertices():
        x, y = v.rotated(ship.body.angle) + ship.body.position
        vs[i][0] = x
        vs[i][1] = y
        i += 1

    ship_patch = patches.Polygon(vs, True)
    # ship_patch = patches.Circle((ship.body.position.x, ship.body.position.y), 0.5)

    for row in list_of_obstacles:
        circles.append(create_circle(space, row[1], row[0], row[2]))
        patch_list.append(patches.Circle((row[1], row[0]), row[2], fill=False))

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
