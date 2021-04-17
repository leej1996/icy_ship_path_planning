import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches
import motion_primitives as mp
import cost_map as cm
from queue import PriorityQueue
import dubins


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


def get_swath(e, n, start_pos, swath_set):
    # heading must be a cardinal direction
    start_pos = np.asarray(start_pos)
    goal_pos = Concat(start_pos,e)
    #print("swath start: ", start_pos, sep=" ")
    #print("swath goal:",goal_pos, sep=" ")
    # check if goal pos is within bounds
    if goal_pos[0] >= n or goal_pos[0] < 0 or goal_pos[1] >= n or goal_pos[1] < 0:
        return False, False
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    swath = np.zeros((n, n))
    # start_pos[2] = goal_pos[2]
    heading = start_pos[2]
    if heading % 90 == 0:
        while not np.array_equal(start_pos[0:2], goal_pos[0:2]):
            if heading == 90 or heading == 270:
                if dx < 0:
                    start_pos[0] = start_pos[0] - 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dx = dx + 1
                elif dx > 0:
                    start_pos[0] = start_pos[0] + 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dx = dx - 1

                if dy < 0:
                    start_pos[1] = start_pos[1] - 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dy = dy + 1
                elif dy > 0:
                    start_pos[1] = start_pos[1] + 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dy = dy - 1
            elif heading == 0 or heading == 180:
                if dy < 0:
                    start_pos[1] = start_pos[1] - 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dy = dy + 1
                elif dy > 0:
                    start_pos[1] = start_pos[1] + 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dy = dy - 1

                if dx < 0:
                    start_pos[0] = start_pos[0] - 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dx = dx + 1
                elif dx > 0:
                    start_pos[0] = start_pos[0] + 1
                    swath[start_pos[1], start_pos[0]] = 1
                    dx = dx - 1
    else:
        # edge_set_ordinal = [(-1,-1,45), (-1, -1, 90), (-3, 0, 135), (-1, -1, 0), (0, -3, 315)]
        if e[2] == 0:
            # straight
            swath1 = swath_set[("straight", heading)]
        elif e[2] == 45:
            # left 45
            swath1 = swath_set[("left_turn_45", heading)]
        elif e[2] == 90:
            # left 90
            swath1 = swath_set[("left_turn", heading)]
        elif e[2] == 315:
            # right 45
            swath1 = swath_set[("right_turn_45", heading)]
        else:
            # right 90
            swath1 = swath_set[("right_turn", heading)]

        swath_size = swath1.shape[0]
        min_y = start_pos[1] - 3
        max_y = start_pos[1] + 4
        min_x = start_pos[0] - 3
        max_x = start_pos[0] + 4
        # Too far to the right
        if start_pos[0] + 4 >= n:
            overhang = start_pos[0] + 4 - (n - 1)
            swath1 = np.delete(swath1, slice(swath_size-overhang,swath_size), axis=1)
            max_x = n - 1
        # Too far to the left
        elif start_pos[0] - 3 < 0:
            overhang = abs(start_pos[0] - 3)
            swath1 = np.delete(swath1, slice(0,overhang), axis=1)
            min_x = 0
        # Too close to the top
        if start_pos[1] + 4 >= n:
            overhang = start_pos[1] + 4 - (n - 1)
            swath1 = np.delete(swath1, slice(swath_size-overhang,swath_size), axis=0)
            # print(slice(swath_size-overhang,swath_size))
            max_y = n - 1
        # Too close to the bottom
        elif start_pos[1] - 3 < 0:
            overhang = abs(start_pos[1] - 3)
            swath1 = np.delete(swath1, slice(0,overhang), axis=0)
            min_y = 0
        # print("test1", min_y,max_y,min_x,max_x,sep=" ")
        # print(swath1)
        swath[min_y:max_y, min_x:max_x] = swath1

    return swath, goal_pos


def Concat(x,y):
    """
    given two points x,y in the lattice, find the concatenation x + y
    """
    p1 = [x[0], x[1], x[2]]
    p2 = [y[0], y[1], y[2]]
    if p1[2] == 0 or p1[2] == 45:
        R = np.eye(3)
    else:
        if p1[2] == 135 or p1[2] == 225 or p1[2] == 315:
            card = 45
        else:
            card = 0
        heading = math.radians(p1[2] - card)
        R = np.asarray([[math.cos(heading), -math.sin(heading),0],[math.sin(heading), math.cos(heading),0],[0,0,1]])
    multiplication = np.matmul(R,np.transpose(np.asarray(p2)))
    result = np.asarray(p1) + multiplication
    heading = result[2]

    while heading >= 360:
        heading = heading - 360

    return (int(result[0]), int(result[1]), round(heading))


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
    #math.radians(P1[2] + 90) % (2*math.pi)
    p1 = (p_initial[0], p_initial[1], math.radians((p_initial[2] + 90) % 360))
    p2 = (p_final[0], p_final[1], math.radians((p_final[2] + 90) % 360))
    path = dubins.shortest_path(p1, p2, turning_radius)
    return path.path_length()


def a_star(start, goal, turning_radius, n, cost_map, card_edge_set, ord_edge_set, swath_set):
    # theta is measured ccw from y axis
    openSet = [start]
    closedSet = []
    openSet.append(start)
    cameFrom = dict()
    cameFrom[start] = None
    cameFrom_by_edge = dict()
    cameFrom_by_edge[start] = None
    # cost from start
    g_score = dict()
    g_score[start] = 0
    # heuristic (estimation of cost to goal)
    f_score = dict()
    f_score[start] = heuristic(start, goal, turning_radius)
    # priority queue of all visited node f scores
    f_score_open_sorted = CustomPriorityQueue()
    f_score_open_sorted.put((start, f_score[start]))
    counter = 0
    while np.shape(openSet)[0] != 0:
        # node[0] = x position
        # node[1] = y position
        # node[2] = theta (heading)
        # node = openSet.pop()
        node = f_score_open_sorted.get()[0]
        print("Counter: ",counter, sep=" ")
        #file.write("***NEW*** node: {0}".format(node))

        if node == goal:
            path = list()
            cameFrom[goal] = cameFrom[node]
            path.append(goal)
            while node != start:
                pred = cameFrom[node]
                node = pred
                path.append(node)
            return (True, f_score[goal], path, closedSet)

        openSet.remove(node)
        closedSet.append(node)

        if node[2] % 90 == 0:
            edge_set = card_edge_set
        else:
            edge_set = ord_edge_set
        for e in edge_set:
            swath, neighbour = get_swath(e, n, np.array(node), swath_set)
            if not neighbour:
                continue

            # check if point is in closed set
            neighbour_in_closed_set, closed_set_neighbour = is_point_in_set(neighbour, closedSet)
            if neighbour_in_closed_set:
                continue

            if 0 < neighbour[0] < n and 0 < neighbour[1] < n:
                # mx = np.ma.masked_array(cost_map, mask=swath)
                #mask = np.zeros_like(swath)
                #mask[swath] = cost_map
                swath = swath.astype(bool)
                mask = cost_map[swath]
                cost = np.sum(mask) + heuristic(node, neighbour, turning_radius)
                #cost = np.sum(mask)
                temp_g_score = g_score[node] + cost
                neighbour_in_open_set, open_set_neighbour = is_point_in_set(neighbour, openSet)
                if not neighbour_in_open_set:
                    openSet.append(neighbour)
                    cameFrom[neighbour] = node
                    cameFrom_by_edge[neighbour] = e
                    g_score[neighbour] = temp_g_score
                    f_score[neighbour] = g_score[neighbour] + heuristic(neighbour, goal, turning_radius)
                    f_score_open_sorted.put((neighbour, f_score[neighbour]))
                elif neighbour_in_open_set and temp_g_score < g_score[open_set_neighbour]:
                    cameFrom[open_set_neighbour] = node
                    cameFrom_by_edge[open_set_neighbour] = e
                    g_score[open_set_neighbour] = temp_g_score
                    f_score_open_sorted._update((open_set_neighbour, f_score[open_set_neighbour]),
                                                g_score[open_set_neighbour] + heuristic(open_set_neighbour, goal,turning_radius))
                    f_score[open_set_neighbour] = g_score[open_set_neighbour] + heuristic(open_set_neighbour, goal,turning_radius)
        counter = counter + 1
    return (False, 'Fail', 'Fail')


def main():
    n = 71
    r = 7  # radius of circular turns
    d = 1  # how far robot goes in straight line
    theta = 0  # measured clockwise from y axis (up)
    turning_radius = 1.99
    scale = 0.5
    start_pos = (math.floor(n / 2), 0, 0)
    # start_pos = (65,0,0)
    #goal_pos = (60, 65, 0)
    goal_pos = (35,60,0)
    # goal_pos = (20,50,90)
    cost_map = np.zeros((n, n))
    # list_of_obstacles = np.array([[12, 12, 10], [25, 25, 8], [38, 36, 4], [25, 55, 15]])
    # list_of_obstacles = np.array([[35,35,15]])
    # list_of_obstacles = np.array([[5,5,5], [20,30,5], [50,40,5], [10,60,5], [40,10,5], [30,30,5],[30,50,5]])
    list_of_obstacles = np.array([[30,5,5], [30,16,5],[30,27,5],[30,38,5],[30,49,5],[30,60,5],[30,68,2]])
    for row in list_of_obstacles:
        cost_map = cm.create_circle(row, cost_map, scale)

    # y is pointing down, x is pointing to the right
    edge_set_cardinal = [(0, 1, 0), (-1, 2, 45), (-2, 2, 90), (1, 2, 315), (2, 2, 270)]
    edge_set_ordinal = [(-1, 1, 0), (-2, 1, 45), (-3, 0, 90), (-1, 2, 315), (0, 3, 270)]

    # swath set
    ordinal_swaths = dict()
    swath_straight_45 = np.zeros((7, 7))
    swath_straight_45[4, 2] = 1
    ordinal_swaths[("straight", 45)] = swath_straight_45

    swath_straight_135 = np.zeros((7, 7))
    swath_straight_135[2, 2] = 1
    ordinal_swaths[("straight", 135)] = swath_straight_135

    swath_straight_225 = np.zeros((7, 7))
    swath_straight_225[2, 4] = 1
    ordinal_swaths[("straight", 225)] = swath_straight_225

    swath_straight_315 = np.zeros((7, 7))
    swath_straight_315[4, 4] = 1
    ordinal_swaths[("straight", 315)] = swath_straight_315

    # [y, x]
    swath_left_turn_45 = np.zeros((7, 7))
    swath_left_turn_45[4, 1:3] = 1
    swath_left_turn_45[3, 0] = 1
    ordinal_swaths[("left_turn", 45)] = swath_left_turn_45

    swath_left_turn_135 = np.zeros((7, 7))
    swath_left_turn_135[1:3, 2] = 1
    swath_left_turn_135[0, 3] = 1
    ordinal_swaths[("left_turn", 135)] = swath_left_turn_135

    swath_left_turn_225 = np.zeros((7, 7))
    swath_left_turn_225[2, 4:6] = 1
    swath_left_turn_225[3, 6] = 1
    ordinal_swaths[("left_turn", 225)] = swath_left_turn_225

    swath_left_turn_315 = np.zeros((7, 7))
    swath_left_turn_315[4:6, 4] = 1
    swath_left_turn_315[6, 3] = 1
    ordinal_swaths[("left_turn", 315)] = swath_left_turn_315

    swath_left_45_turn_45 = np.zeros((7, 7))
    swath_left_45_turn_45[4, 1:3] = 1
    ordinal_swaths[("left_turn_45", 45)] = swath_left_45_turn_45

    swath_left_45_turn_135 = np.zeros((7, 7))
    swath_left_45_turn_135[1:3, 2] = 1
    ordinal_swaths[("left_turn_45", 135)] = swath_left_45_turn_135

    swath_left_45_turn_225 = np.zeros((7, 7))
    swath_left_45_turn_225[2, 4:6] = 1
    ordinal_swaths[("left_turn_45", 225)] = swath_left_45_turn_225

    swath_left_45_turn_315 = np.zeros((7, 7))
    swath_left_45_turn_315[4:6, 4] = 1
    ordinal_swaths[("left_turn_45", 315)] = swath_left_45_turn_315

    swath_right_turn_45 = np.zeros((7, 7))
    swath_right_turn_45[4:6, 2] = 1
    swath_right_turn_45[6, 3] = 1
    ordinal_swaths[("right_turn", 45)] = swath_right_turn_45

    swath_right_turn_135 = np.zeros((7, 7))
    swath_right_turn_135[2, 1:3] = 1
    swath_right_turn_135[3, 0] = 1
    ordinal_swaths[("right_turn", 135)] = swath_right_turn_135

    swath_right_turn_225 = np.zeros((7, 7))
    swath_right_turn_225[1:3, 4] = 1
    swath_right_turn_225[0, 3] = 1
    ordinal_swaths[("right_turn", 225)] = swath_right_turn_225

    swath_right_turn_315 = np.zeros((7, 7))
    swath_right_turn_315[4, 4:6] = 1
    swath_right_turn_315[3, 6] = 1
    ordinal_swaths[("right_turn", 315)] = swath_right_turn_315

    swath_right_45_turn_45 = np.zeros((7, 7))
    swath_right_45_turn_45[4:6, 2] = 1
    ordinal_swaths[("right_turn_45", 45)] = swath_right_45_turn_45

    swath_right_45_turn_135 = np.zeros((7, 7))
    swath_right_45_turn_135[2, 1:3] = 1
    ordinal_swaths[("right_turn_45", 135)] = swath_right_45_turn_135

    swath_right_45_turn_225 = np.zeros((7, 7))
    swath_right_45_turn_225[1:3, 4] = 1
    ordinal_swaths[("right_turn_45", 225)] = swath_right_45_turn_225

    swath_right_45_turn_315 = np.zeros((7, 7))
    swath_right_45_turn_315[4, 4:6] = 1
    ordinal_swaths[("right_turn_45", 315)] = swath_right_45_turn_315

    '''
    swath, neighbour = get_swath(edge_set_cardinal[3], n, start_pos, ordinal_swaths)
    swath = swath.astype(bool)
    print(swath)
    #ax.imshow(swath, origin='lower')
    mask = cost_map[swath]
    print(mask)
    print(np.sum(mask))
    ax.imshow(swath,origin='lower')
    plt.show()
    '''


    # start_pos_test = (math.floor(n / 2), math.floor(n / 2))

    Worked, L, Edge_path1, nodes_visited= a_star(start_pos, goal_pos, turning_radius, n, cost_map, edge_set_cardinal, edge_set_ordinal, ordinal_swaths)
    # Worked = False
    #node_plot = np.zeros((n,n))
    #for node in nodes_visited:
    #    node_plot[node[1], node[0]] = node_plot[node[1], node[0]] + 1
    #np.savetxt("nodes.csv", node_plot, delimiter=",")

    fig, ax = plt.subplots(2,1)

    if Worked:
        ax[0].imshow(cost_map, origin='lower')
        xmax = 0
        ymax = 0
        PATH = [i for i in Edge_path1[::-1]]
        array = np.zeros((2,1))
        for i in range(np.shape(PATH)[0] - 1):
            P1 = PATH[i]
            P2 = PATH[i + 1]
            math.radians(start_pos[2] + 90) % (2*math.pi)
            path = dubins.shortest_path((P1[0], P1[1], math.radians(P1[2] + 90) % (2*math.pi)), (P2[0], P2[1], math.radians(P2[2] + 90) % (2*math.pi)), turning_radius)
            configurations, _ =path.sample_many(0.2)
            #0.01
            x = list()
            y = list()
            for config in configurations:
                x.append(config[0])
                y.append(config[1])
                if config[0] > xmax:
                    xmax = config[0]
                if config[1] > ymax:
                    ymax = config[1]
            ax[0].plot(x, y, 'g')
            #print(np.array([np.asarray(x).T, np.asarray(y).T]))
            array = np.append(array, np.array([np.asarray(x).T, np.asarray(y).T]), axis=1)
            print(np.shape(array))

        #print(y)
        #array = np.array([np.asarray(x).T, np.asarray(y).T])
        array = np.delete(array, 0, 1)
        print(np.shape(array))
        np.savetxt("nodes.csv", array, delimiter=",")
        #print(array)
        #file1 = open("log.txt","w")
        #for x1, y1 in zip(x,y):
        #    print(x1, y1, sep=",",file=file1)
        #file1.close()
        for obs in list_of_obstacles:
            ax[0].add_patch(patches.Circle((obs[1],obs[0]),obs[2],fill=False))

    node_plot = np.zeros((n,n))
    for node in nodes_visited:
        node_plot[node[1], node[0]] = node_plot[node[1], node[0]] + 1
    #np.savetxt("nodes.csv", node_plot, delimiter=",")
    ax[1].imshow(node_plot, origin='lower')
    plt.show()
    print("Total Cost:", L, sep=" ")


if __name__ == "__main__":
    main()
