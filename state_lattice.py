import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
import cost_map as cm
import pymunk
from pymunk.vec2d import Vec2d
from queue import PriorityQueue
import copy
import dubins
import time
import random


class CustomPriorityQueue(PriorityQueue):
    def _put(self, item):
        return super()._put((self._get_priority(item), item))

    def _get(self):
        return super()._get()[1]

    def _get_priority(self, item):
        return item[1]

    def _update(self, item, update_value):
        self.queue.remove(((item[1]), (item[0], item[1])))
        self._put((item[0], update_value))


def generate_obstacles(start, goal, n, m, num_obs, min_r, max_r):
    obstacles = list()
    #for obs in range(num_obs):
    protection = 0
    x = random.randint(max_r, m - max_r)
    y = random.randint(start[1] + 5 + max_r, goal[1] - 5 - max_r)
    r = random.randint(min_r, max_r)
    obstacles.append([y, x, r])
    while len(obstacles) < num_obs - 1:
    #for i in range(num_obs - 1):
        near_obs = False
        x = random.randint(max_r, m - max_r - 1)
        y = random.randint(start[1] + 5 + max_r, goal[1] - 5 - max_r)
        r = random.randint(min_r, max_r)
        print(x,y)
        # check if obstacles overlap
        for obs in obstacles:
            #((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            print("print",obs[1], obs[0])
            distance = ((x - obs[1]) ** 2 + (y - obs[0]) ** 2) ** 0.5
            print(distance)
            if ((x - obs[1]) ** 2 + (y - obs[0]) ** 2) ** 0.5 < 2 * max_r:
                print('near obstacle')
                near_obs = True
                continue
        if not near_obs:
            obstacles.append([y, x, r])

        protection += 1

        if protection > 300:
            break


    print(obstacles)
    return obstacles


'''
def get_cost(start, edge, turning_radius, list_of_obstacles):
    R = np.asarray([[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0], [math.sin(math.pi / 2), math.cos(math.pi / 2), 0],
                    [0, 0, 1]])
    rot_e = R @ np.asarray(edge).T + np.array([start[0], start[1], 0])
    dubins_path = dubins.shortest_path((start[0], start[1], math.radians((start[2] + 2) * 45)),
                                        (rot_e[0], rot_e[1], math.radians((rot_e[2] + 2) * 45) % (2 * math.pi)),
                                        turning_radius)
    configurations, _ = dubins_path.sample_many(0.1)
    check = True
    min_dist = 0
    for obs in list_of_obstacles:
        # check if ship is within radius + 5 squares of the center of obstacle, then do swath
        #determine distance from edge of obstacle
        temp = dist(configurations[0], (obs[1],obs[0])) - obs[2]
        min_dist = min(temp, min_dist)

    if min_dist < 1:
        check = True
    else:
        check = False

    for config in configurations:
        # (y,x)
        if check:
            
        else:
            if min_dist > 0:
                min_dist = min_dist - 0.1
                continue
            else:
                check = True
        x_cell = int(round(config[0]))
        y_cell = int(round(config[1]))
        if [x_cell, y_cell] not in swath:
            swath.append([x_cell, y_cell])

'''
def generate_swath(edge_set, turning_radius, heading):
    '''
    Will have key of (edge, start heading)
    '''
    swath_set = dict()
    R = np.asarray(
        [[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0], [math.sin(math.pi / 2), math.cos(math.pi / 2), 0],
         [0, 0, 1]])
    start_pos = (5, 5, heading)
    # fig = plt.figure()
    # ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
    # ax.set_aspect("equal")
    for e in edge_set:
        array = np.zeros((11, 11), dtype=bool)
        swath = [[start_pos[0], start_pos[1]]]
        rot_e = R @ np.asarray(e).T + np.array([start_pos[0], start_pos[1], 0])
        dubins_path = dubins.shortest_path((start_pos[0], start_pos[1], math.radians((start_pos[2] + 2) * 45)),
                                           (rot_e[0], rot_e[1], math.radians((rot_e[2] + 2) * 45) % (2 * math.pi)),
                                           turning_radius)
        configurations, _ = dubins_path.sample_many(0.01)
        x = list()
        y = list()
        for config in configurations:
            x.append(config[0])
            y.append(config[1])
            x_cell = int(round(config[0]))
            y_cell = int(round(config[1]))
            if [x_cell, y_cell] not in swath:
                swath.append([x_cell, y_cell])

        for pair in swath:
            array[pair[1], pair[0]] = True
        # ax.plot(x,y)
        swath_set[e, 0 + heading] = array
        swath_set[e, 2 + heading] = np.flip(array.T, 1)  # Rotate 90 degrees CCW
        swath_set[e, 4 + heading] = np.flip(np.flip(array, 1), 0)  # Rotate 180 degrees CCW
        swath_set[e, 6 + heading] = np.flip(array.T, 0)  # Rotate 270 degrees CCW

    return swath_set


def get_swath(e, n, b, start_pos, swath_set):
    #print("start", start_pos, sep=" ")
    #print("edge", e, sep=" ")
    swath = np.zeros((n, b), dtype=bool)
    heading = int(start_pos[2])
    #print("heading", heading, sep=" ")
    swath1 = swath_set[e, heading]

    swath_size = swath1.shape[0]
    min_y = start_pos[1] - 5
    max_y = start_pos[1] + 6
    min_x = start_pos[0] - 5
    max_x = start_pos[0] + 6
    # print(min_y, max_y, min_x, max_x, "overhangs", sep=" ")
    # Too far to the right
    if max_x >= b:
        overhang = max_x - (b - 1)
        swath1 = np.delete(swath1, slice(swath_size - overhang, swath_size), axis=1)
        max_x = b - 1
    # Too far to the left
    elif min_x < 0:
        overhang = abs(min_x)
        swath1 = np.delete(swath1, slice(0, overhang), axis=1)
        min_x = 0
    # Too close to the top
    if max_y >= n:
        overhang = max_y - (n - 1)
        swath1 = np.delete(swath1, slice(swath_size - overhang, swath_size), axis=0)
        max_y = n - 1
    # Too close to the bottom
    elif min_y < 0:
        overhang = abs(min_y)
        swath1 = np.delete(swath1, slice(0, overhang), axis=0)
        min_y = 0
    swath[min_y:max_y, min_x:max_x] = swath1

    return swath


def Concat(x, y):
    """
    given two points x,y in the lattice, find the concatenation x + y
    """

    #p1 = [x[0], x[1], x[2] * math.pi / 4]
    rot = x[2] * math.pi / 4
    #print("original rot", x[2])
    p1 = [x[0], x[1]]
    p2_theta = y[2] * math.pi / 4
    p2 = [y[0], y[1]]

    if x[2] % 2 == 0:
        #cardinal
        heading = p2_theta + rot
    else:
        #ordinal
        rot = rot - math.pi/4
        heading = p2_theta + rot

    #R = np.asarray([[math.cos(rot + math.pi / 2), -math.sin(rot + math.pi / 2), 0],
    #                [math.sin(rot + math.pi / 2), math.cos(rot + math.pi / 2), 0], [0, 0, 1]])
    R = np.array([[math.cos(rot + math.pi/2), -math.sin(rot + math.pi/2)],
                  [math.sin(rot + math.pi/2), math.cos(rot + math.pi/2)]])
    multiplication = np.matmul(R, np.transpose(np.asarray(p2)))
    #print("rot", rot/(math.pi/4))
    #print(multiplication)
    result = np.asarray(p1) + multiplication

    while heading >= 2 * math.pi:
        heading = heading - 2 * math.pi
    heading = heading / (math.pi / 4)

    return (int(result[0]), int(result[1]), int(round(heading)))


def is_point_in_set(point, set):
    for point1 in set:
        if dist(point, point1[0]) < 0.001 and abs(point[2] - point1[0][2]) < 0.001:
            return True, point1[0], point1[1]
    return False, False, False


def is_point_in_set_euclid(point,set):
    for point1 in set:
        if dist(point, point1) < 0.001:
            return True, point1
    return False, False


def is_node_in_gen_cap(generation, m, node, set):
    '''
    Go through last m generations and check if node was found in that time
    '''
    for gen in range(0,m):
        temp_copy = copy.deepcopy(set[generation - gen])
        while not temp_copy.empty():
            if node == temp_copy.get():
                return True
    return False


def dist(a, b):
    # Euclidean distance
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def heuristic(p_initial, p_final, turning_radius, n, b, cost_map):
    """
    The Dubins' distance from initial to final points.
    """
    '''
    start = (p_initial[0],p_initial[1])
    goal = (p_final[0], p_final[1])
    print("start", p_initial, sep=" ")
    print("goal", p_final, sep=" ")
    Worked, euclid_cost = a_star_euclid(start, goal, n, b, cost_map)
    print("euclid:", euclid_cost, sep=" ")
    '''

    dubins_cost = dubins_shortest_path(p_initial,p_final,turning_radius)
    #print("dubins:", dubins_cost, sep=" ")
    #return max(dubins_cost, euclid_cost)
    return dubins_cost


def dubins_shortest_path(p_initial, p_final, turning_radius):
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
        if dist(node, (obs[1],obs[0])) < obs[2] + 5 and not past_obstacle(node, obs):
            return True
    return False


def past_obstacle(node, obs):
    # also check if ship is past all obstacles (under assumption that goal is always positive y direction from start)
    # obs y coord + radius
    if node[1] > obs[0] + obs[2]:
        return True
    else:
        return False

'''
def recursion(n,m,cost_map, start_pos, goal_pos):
    dp = np.zeros((n*m,n*m))
    for x1 in range(n):
        for y1 in range(m):
'''

def a_star_euclid(start, goal, n, b, cost_map):
    openSet = [start]
    closedSet = []
    cameFrom = dict()
    cameFrom[start] = None
    # cost from start
    g_score = dict()
    g_score[start] = 0
    # heuristic (estimation of cost to goal)
    f_score = dict()
    f_score[start] = dist(start, goal)
    # priority queue of all visited node f scores
    f_score_open_sorted = CustomPriorityQueue()
    f_score_open_sorted.put((start, f_score[start]))

    while np.shape(openSet)[0] != 0:
        # node[0] = x position
        # node[1] = y position
        # node[2] = theta (heading)

        node = f_score_open_sorted.get()[0]
        if node == goal:
            path = list()
            cameFrom[goal] = cameFrom[node]
            path.append(goal)
            while node != start:
                pred = cameFrom[node]
                node = pred
                path.append(node)
            return (True, f_score[goal])

        openSet.remove(node)
        closedSet.append(node)

        x = node[0]
        y = node[1]
        #(pm1, 0), (pm3, pm1), (pm2, pm1), (pm1, pm1), (pm1, pm2), (pm1, pm3), (0, pm1),
        neighbour_list = [(x + 3, y + 1), (x - 3, y + 1), (x + 3, y - 1), (x - 3, y - 1),
                          (x + 2, y + 1), (x - 2, y + 1), (x + 2, y - 1), (x - 2, y - 1),
                          (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2),
                          (x + 1, y + 3), (x + 1, y - 3), (x - 1, y + 3), (x - 1, y - 3),
                          (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),(x - 1, y - 1),
                          (x, y + 1), (x + 1, y),(x - 1, y), (x, y - 1)]
        for neighbour in neighbour_list:
            if 0 <= neighbour[0] < b and 0 <= neighbour[1] < n:
                # check if point is in closed set
                neighbour_in_closed_set, closed_set_neighbour, = is_point_in_set_euclid(neighbour, closedSet)
                if neighbour_in_closed_set:
                    #print("hello")
                    continue

                swath_cost = cost_map[neighbour[1],neighbour[0]]
                cost = swath_cost + dist(node, neighbour)
                temp_g_score = g_score[node] + cost
                neighbour_in_open_set, open_set_neighbour = is_point_in_set_euclid(neighbour, openSet)
                if not neighbour_in_open_set:
                    openSet.append(neighbour)
                    cameFrom[neighbour] = node
                    g_score[neighbour] = temp_g_score
                    f_score[neighbour] = g_score[neighbour] + dist(neighbour, goal)
                    f_score_open_sorted.put((neighbour, f_score[neighbour]))
                elif neighbour_in_open_set and temp_g_score < g_score[open_set_neighbour]:
                    cameFrom[open_set_neighbour] = node
                    g_score[open_set_neighbour] = temp_g_score
                    f_score_open_sorted._update((open_set_neighbour, f_score[open_set_neighbour]),
                                                 g_score[open_set_neighbour] + dist(open_set_neighbour, goal))
                    f_score[open_set_neighbour] = g_score[open_set_neighbour] + dist(open_set_neighbour, goal)

    return (False, 'Fail',)


def path_smoothing(path, path_length, cost_map, turning_radius, start, goal, nodes, n, m):
    print("Attempt Smoothing")
    #'''
    
    total_length = np.sum(path_length)
    probabilities = np.asarray(path_length)/total_length
    print(len(path))
    #fig, ax = plt.subplots()
    #ax2 = plt.axes(xlim=(0, b), ylim=(0, n))
    #ax.set_aspect("equal")
    x = list()
    y = list()

    for vi in path:
        x.append(vi[0])
        y.append(vi[1])

    #print(probabilities)
    segments = np.sort(np.random.choice(np.arange(len(path)), nodes, p=probabilities))
    #print(segments)

    added_x = list()
    added_y = list()
    counter = 0
    offset = 0
    while counter < len(segments):
        #print("counter", counter)
        node_id = segments[counter]
        #print("node_id", node_id)
        num_values = np.shape(segments[segments == node_id])[0]
        node = path[node_id + offset]
        prev_node = path[node_id + offset - 1]
        #print(prev_node)
        #print(node)

        prim = dubins.shortest_path((prev_node[0], prev_node[1], math.radians((prev_node[2] + 2) * 45)),
                                    (node[0], node[1], math.radians((node[2] + 2) * 45)), turning_radius)
        configurations, _ = prim.sample_many(0.1)
        #seperator = len(configurations)/(num_values + 1)
        #print(np.linspace(0, len(configurations), num=num_values + 2, endpoint = False))
        values = [configurations[int(i)] for i in np.linspace(0, len(configurations), num=num_values + 2, endpoint=False)]
        values.pop(0)
        values.pop()

        inc = 1
        for v in values:
            heading = v[2] - math.pi/2
            added_x.append(v[0])
            added_y.append(v[1])
            if heading < 0:
                heading = heading + 2 * math.pi

            path.insert(node_id - 1 + inc + counter, (v[0],v[1],heading/(math.pi/4)))
            inc += 1
        counter = counter + num_values
        offset = offset + inc - 1

    #print(path)

    #plt.plot(x,y, 'bo')
    #plt.plot(added_x,added_y,'go')
    #'''

    prev = dict()
    smooth_cost = dict()
    for vi in path:
        smooth_cost[vi] = math.inf
        prev[vi] = None
    smooth_cost[path[0]] = 0

    i = 0
    for vi in path:
        for vj in path[i + 1:]:
            # determine cost between node vi and vj
            swath = [[vi[0], vi[1]]]
            swath_cost = 0
            dubins_path = dubins.shortest_path((vi[0], vi[1], math.radians((vi[2] + 2) * 45) % (2 * math.pi)),
                                               (vj[0], vj[1], math.radians((vj[2] + 2) * 45) % (2 * math.pi)),
                                               turning_radius)
            configurations, _ = dubins_path.sample_many(0.9)
            #print(len(configurations))
            #t0 = time.clock()
            for config in configurations:
                x_cell = int(round(config[0]))
                y_cell = int(round(config[1]))
                if 0 <= x_cell < m and 0 <= y_cell < n:
                    if [x_cell, y_cell] not in swath:
                        swath.append([x_cell, y_cell])
                        swath_cost += cost_map[y_cell, x_cell]
            #t1 = time.clock() - t0
            #print("check swath", t1)
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
    #print("length", len(smooth_path), sep=" ")
    #print(smooth_path)
    return smooth_path, x, y, added_x, added_y


def a_star(start, goal, turning_radius, n, m, cost_map, card_edge_set, ord_edge_set, cardinal_swath, ordinal_swath, list_of_obstacles, g):
    # theta is measured ccw from y axis
    a = 0.3
    b = 0.7
    free_path_interval = 5
    generation = 0
    #openSet = [(start,generation)]
    openSet = dict()
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
    f_score[start] = heuristic(start, goal, turning_radius, n, b, cost_map)
    # path length between nodes
    path_length = dict()
    path_length[start] = 0
    # priority queue of all visited node f scores
    f_score_open_sorted = dict()
    f_score_open_sorted[generation] = CustomPriorityQueue()
    f_score_open_sorted[generation].put((start, f_score[start]))
    f_score_open_sorted_nogen = CustomPriorityQueue()
    f_score_open_sorted_nogen.put(((start,generation),f_score[start]))
    # temp dict to keep g generations
    temp = CustomPriorityQueue()
    temp.put(((start, generation), f_score[start]))

    #while np.shape(openSet)[0] != 0:
    while len(openSet) != 0:
        # node[0] = x position
        # node[1] = y position
        # node[2] = theta (heading)

        #node = f_score_open_sorted[generation].get()[0]
        #t0 = time.clock()
        if temp.empty():
            new_node = f_score_open_sorted_nogen.get()[0]
            print("empty temp",new_node, sep=" ")
            temp.put(new_node)
            node, node_gen = temp.get()
            f_score_open_sorted[node_gen].get()
        else:
            node, node_gen = temp.get()[0]
            f_score_open_sorted[node_gen].get()
            f_score_open_sorted_nogen.queue.remove(((f_score[node]), ((node, node_gen), f_score[node])))
        #t1 = time.clock() - t0
        #print("TEMP empty time", t1)
        #print("Generation: ", generation, sep=" ")
        #print("NODE:", node, sep=" ")
        #print("NODE generation", node_gen)
        #print("temp length:", temp.qsize(), sep=" ")

        if generation % g == 0 and generation != 0:
            # Clear temp queue
            while not temp.empty():
                try:
                    temp.get(False)
                except temp.empty():
                    continue

        #if generation > 33:
            #print("start debug")
        # If ship past all obstacles, calc path to goal
        if generation % 5 == 0:
            past_all_obs = True
            for obs in list_of_obstacles:
                if not past_obstacle(node, obs):
                    past_all_obs = False
                    break

            if past_all_obs:
                print("Found path to goal")
                cameFrom[goal] = node
                path_length[goal] = dubins_shortest_path(node, goal, turning_radius)
                f_score[goal] = g_score[node] + path_length[goal]
                node = goal

        if node == goal:
            #print("Found path")
            path = list()
            new_path_length = list()
            cameFrom[goal] = cameFrom[node]
            path.append(goal)
            new_path_length.append(path_length[goal])
            while node != start:
                pred = cameFrom[node]
                node = pred
                path.append(node)
                new_path_length.append(path_length[node])

            #smooth_path = path
            #total_length = np.sum(new_path_length)
            #probabilities = np.asarray(new_path_length)/total_length

            #'''
            path.reverse()  # path: start -> goal
            new_path_length.reverse()
            add_nodes = int(len(path)/3)
            #add_nodes = 0
            #print(add_nodes)
            t0 = time.clock()
            smooth_path, x1, y1, x2, y2 = path_smoothing(path, new_path_length, cost_map, turning_radius, start, goal, add_nodes, n, m)
            t1 = time.clock() - t0
            print("smooth time", t1)
            #'''
            #return (True, f_score[goal], smooth_path, closedSet, x1, y1, x2, y2)
            return (True, f_score[goal], smooth_path, closedSet)

        openSet.pop(node)
        closedSet.append((node, node_gen))

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

            if not neighbour:
                continue

            if 0 <= neighbour[0] < m and 0 <= neighbour[1] < n:
                # check if point is in closed set
                neighbour_in_closed_set, closed_set_neighbour, _, = is_point_in_set(neighbour, closedSet)
                if neighbour_in_closed_set:
                    continue

                if near_obstacle(node, list_of_obstacles):
                    #print("calculate_swath")
                    #t0 = time.clock()
                    swath = get_swath(e, n, m, np.array(node), swath_set)
                    #t2 = time.clock() - t0
                    #print("get_swath time", t2)
                    #t7 = time.clock()
                    swath = swath.astype(bool)
                    #t8 = time.clock() - t7
                    #print("bool conversion time", t8)
                    #t3 = time.clock()
                    mask = cost_map[swath]
                    #t4 = time.clock() - t3
                    #print("mask conversion time", t4)
                    #print(mask)
                    #t5 = time.clock()
                    swath_cost = np.sum(mask)
                    #t6 = time.clock() - t5
                    #print("sum time", t6)
                    #t1 = time.clock() - t0
                    #print("swath time", t1)
                    #print("swath cost", swath_cost)
                else:
                    swath_cost = 0

                #t0 = time.clock()
                temp_path_length = dubins_shortest_path(node, neighbour, turning_radius)
                cost = swath_cost + temp_path_length
                #t1 = time.clock() - t0
                #print("cost time", t1)
                # cost[(node, neighbour)] = np.sum(mask) + heuristic(node, neighbour, turning_radius)
                #print("cost", cost)
                #t0 = time.clock()
                temp_g_score = g_score[node] + cost
                #neighbour_in_open_set, open_set_neighbour, open_set_gen = is_point_in_set(neighbour, openSet)

                if neighbour in openSet:
                    neighbour_in_open_set = True
                    open_set_neighbour = neighbour
                    open_set_gen = openSet[neighbour]
                else:
                    neighbour_in_open_set = False
                    open_set_neighbour = False
                    open_set_gen = False
                #t1 = time.clock() - t0
                #print("OPEN SET time", t1)
                #print(len(openSet))

                #t0 = time.clock()
                if not neighbour_in_open_set:
                    #print("new node")
                    heuristic_value = heuristic(neighbour, goal, turning_radius, n, m, cost_map)
                    #openSet.append((neighbour, generation))
                    openSet[neighbour] = generation
                    cameFrom[neighbour] = node
                    cameFrom_by_edge[neighbour] = e
                    path_length[neighbour] = temp_path_length
                    g_score[neighbour] = temp_g_score
                    f_score[neighbour] = a * g_score[neighbour] + b * heuristic_value
                    f_score_open_sorted[generation].put((neighbour, f_score[neighbour]))
                    temp.put(((neighbour, generation), f_score[neighbour]))
                    f_score_open_sorted_nogen.put(((neighbour, generation), f_score[neighbour]))
                elif neighbour_in_open_set and temp_g_score < g_score[open_set_neighbour]:
                    open_set_neighbour_heuristic_value = heuristic(open_set_neighbour, goal, turning_radius, n, m, cost_map)
                    #print("found cheaper cost to node")
                    #print(open_set_neighbour)
                    #print(open_set_gen)
                    cameFrom[open_set_neighbour] = node
                    cameFrom_by_edge[open_set_neighbour] = e
                    path_length[open_set_neighbour] = temp_path_length
                    g_score[open_set_neighbour] = temp_g_score
                    f_score_open_sorted[open_set_gen]._update((open_set_neighbour, f_score[open_set_neighbour]),
                                                               a * g_score[open_set_neighbour] + b * open_set_neighbour_heuristic_value)
                    f_score_open_sorted_nogen._update(((open_set_neighbour, open_set_gen), f_score[open_set_neighbour]),
                                                        a * g_score[open_set_neighbour] + b * open_set_neighbour_heuristic_value)
                    if generation - open_set_gen < generation % g:
                        #print("hello")
                        temp._update(((open_set_neighbour,open_set_gen), f_score[open_set_neighbour]),
                                     a * g_score[open_set_neighbour] + b * open_set_neighbour_heuristic_value)
                    f_score[open_set_neighbour] = a * g_score[open_set_neighbour] + b * open_set_neighbour_heuristic_value
                #t1 = time.clock() - t0
                #print("set assignment", t1)
        generation = generation + 1
        f_score_open_sorted[generation] = CustomPriorityQueue()
    return (False, 'Fail', 'Fail', 'Fail')


def create_circle(space, x, y, r):
    # body = pymunk.Body(1, 100, body_type=pymunk.Body.DYNAMIC)
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Circle(body, r)
    shape.density = 3
    space.add(body, shape)
    return shape


class Ship:
    def __init__(self, space, v, x, y, theta):
        self.vertices = [(0, 2), (0.5, 1), (0.5, -1), (-0.5, -1), (-0.5, 1)]
        # [(0,3), (1,1), (1,-2), (-1,-2), (-1,1)]
        self.body = pymunk.Body(1, 100, body_type=pymunk.Body.KINEMATIC)
        self.body.position = (x, y)
        self.body.velocity = v
        self.body.angle = math.radians(theta)
        self.shape = pymunk.Poly(self.body, self.vertices)

        # pymunk.Circle(self.body, r)
        space.add(self.body, self.shape)
        self.path_pos = 0

    def set_path_pos(self, path_pos):
        self.path_pos = path_pos


def main():
    # resolution is 50 cm x 50 cm
    # ice tank is 76 x 12 m
    n = 152
    b = 24
    r = 7  # radius of circular turns
    theta = 0  # Possible values: 0 - 7, each number should be multiplied by 45 degrees (measured CCW from up)
    turning_radius = 0.999
    scale = 3
    m = 4
    start_pos = (15, 15, theta)
    # start_pos = (65,0,0)
    # goal_pos = (60, 65, 0)
    goal_pos = (15, 120, 0)
    # goal_pos = (20,50,90)
    cost_map = np.zeros((n, b))
    #val = b/2
    #for i in range(int(b/2)):
        #cost_map[:, i] = val
        #cost_map[:, b-1-i] = val
        #val -= 1
    #plt.imshow(cost_map)
    #plt.show()
    # list_of_obstacles = np.array([[12, 12, 10], [25, 25, 8], [38, 36, 4], [25, 55, 15]])
    # list_of_obstacles = np.array([[35,15,10]])
    # list_of_obstacles = np.array([[5,5,5], [20,30,5], [50,40,5], [10,60,5], [40,10,5], [30,30,5],[30,50,5]])
    #list_of_obstacles = np.array([[30, 5, 5], [30, 16, 5], [30, 27, 5], [30, 38, 5], [30, 49, 5], [30, 60, 5]])
    #list_of_obstacles = np.array([[30, 5, 5], [28, 17, 5], [30, 28, 4], [40,10,5], [50, 6, 5], [50, 20, 5]])
    #list_of_obstacles = np.array([[30, 5, 5], [28, 17, 5], [30, 28, 4], [40,10,5], [50, 6, 5], [50, 20, 5]])
    #list_of_obstacles = list()
    list_of_obstacles = generate_obstacles(start_pos, goal_pos, n, b, 12, 2, 5)
    #list_of_obstacles = [[36, 15, 4], [106, 8, 4], [54, 10, 3], [69, 7, 3], [83, 15, 4]]
    for row in list_of_obstacles:
        cost_map = cm.create_circle(row, cost_map, scale)

    #plt.imshow(cost_map, origin='lower')
    #plt.show()
    # y is pointing up, x is pointing to the right
    # must rotate all swaths pi/4 CCW to be facing up
    edge_set_cardinal = [(1, 0, 0),
    #                     (5, 0, 0),
                         (2, 1, 0),
                         (2, -1, 0),
                         (2, 1, 1),
                         (2, -1, 7),
                         (2, 2, 1),
                         (2, -2, 7),
                         (3, 0, 1),
                         (3, 0, 7),
                         (3, 2, 0),
                         (3, -2, 0),
                         (3, 3, 2),
                         (3, -3, 6),
                         (3, 4, 2),
                         (3, -4, 6),
                         (3, 5, 2),
                         (3, -5, 6),
                         (4, 5, 0),
                         (4, -5, 0)]
    edge_set_ordinal = [(0, 3, 2),
    #                    (3, 3, 1),
                        (0, 4, 3),
                        (0, 5, 1),
                        (0, 5, 3),
                        (1, 1, 1),
                        (1, 2, 1),
                        (1, 2, 2),
                        (1, 3, 1),
                        (1, 4, 1),
                        (2, 1, 0),
                        (2, 1, 1),
                        (2, 2, 0),
                        (2, 2, 2),
                        (3, 0, 0),
                        (3, 1, 1),
                        (4, 0, 7),
                        (4, 1, 1),
                        (5, 0, 1),
                        (5, 0, 7)]

    ordinal_swaths = generate_swath(edge_set_ordinal, turning_radius, 1)
    cardinal_swaths = generate_swath(edge_set_cardinal, turning_radius, 0)

    t0 = time.clock()
    #worked, L, edge_path, nodes_visited, x1, y1, x2, y2 = a_star(start_pos, goal_pos, turning_radius, n, b, cost_map, edge_set_cardinal,
    #                                             edge_set_ordinal, cardinal_swaths, ordinal_swaths, list_of_obstacles, m)
    worked, L, edge_path, nodes_visited = a_star(start_pos, goal_pos, turning_radius, n, b, cost_map, edge_set_cardinal,
                                                 edge_set_ordinal, cardinal_swaths, ordinal_swaths, list_of_obstacles, m)
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    print("Hz", 1/t1)

    fig1, ax1 = plt.subplots(2, 1)
    #fig1, ax1 = plt.subplots()
    '''
    start_pos = (math.floor(n / 2), 10)
    goal_pos = (35, 60)
    worked, L, edge_path = a_star_euclid(start_pos,goal_pos,n,cost_map)

    if worked:
        ax1[0].imshow(cost_map, origin='lower')
        PATH = [i for i in edge_path[::-1]]
        #path = np.zeros((2, 1))
        for i in range(np.shape(PATH)[0] - 1):
            P1 = PATH[i]
            print(P1)
            P2 = PATH[i + 1]
            ax1[0].plot([P1[0],P2[0]], [P1[1],P2[1]], 'g')

    plt.show()
    '''
    #'''
    if worked:
        ax1[0].imshow(cost_map, origin='lower')
        #ax1.imshow(cost_map, origin='lower')
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
            #ax1.plot(x, y, 'g')
            path = np.append(path, np.array([np.asarray(x).T, np.asarray(y).T]), axis=1)

        path = np.delete(path, 0, 1)
        print(np.shape(path))

        for obs in list_of_obstacles:
            ax1[0].add_patch(patches.Circle((obs[1], obs[0]), obs[2], fill=False))
            #ax1.add_patch(patches.Circle((obs[1], obs[0]), obs[2]+0.5, fill=False))
        #ax1.plot(x1,y1, 'bx')
        #ax1.plot(x2,y2,'gx')
    else:
        path = 0

    node_plot = np.zeros((n, b))
    
    for node in nodes_visited:
        #print(node)
        node_plot[node[0][1], node[0][0]] = node_plot[node[0][1], node[0][0]] + 1
    

    ax1[1].imshow(node_plot, origin='lower')
    #'''
    #'''
    print("Total Cost:", L, sep=" ")
    print("Num of nodes expanded",np.sum(node_plot))

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

    for row in list_of_obstacles:
        # circles.append(create_circle(space, row[0], row[1], row[2]))
        # patch_list.append(patches.Circle((row[0], row[1]), row[2], fill=False))
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
    ax2 = plt.axes(xlim=(0, b), ylim=(0, n))
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
    
    #'''


if __name__ == "__main__":
    main()
