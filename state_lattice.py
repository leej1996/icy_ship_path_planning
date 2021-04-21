import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
import cost_map as cm
import pymunk
from pymunk.vec2d import Vec2d
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

def get_ordinal_swaths():
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

    return ordinal_swaths


def get_swath(e, n, start_pos, swath_set):
    start_pos = np.asarray(start_pos)
    goal_pos = Concat(start_pos,e)
    # check if goal pos is within bounds
    if goal_pos[0] >= n or goal_pos[0] < 0 or goal_pos[1] >= n or goal_pos[1] < 0:
        return False, False
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    swath = np.zeros((n, n))
    heading = start_pos[2]
    if heading % 90 == 0:
        # Cardinal Swaths
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
        # Ordinal Swaths
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
            max_y = n - 1
        # Too close to the bottom
        elif start_pos[1] - 3 < 0:
            overhang = abs(start_pos[1] - 3)
            swath1 = np.delete(swath1, slice(0,overhang), axis=0)
            min_y = 0
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
                swath = swath.astype(bool)
                mask = cost_map[swath]
                cost = np.sum(mask) + heuristic(node, neighbour, turning_radius)
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


def create_circle(space, x, y, r):
    #body = pymunk.Body(1, 100, body_type=pymunk.Body.DYNAMIC)
    body = pymunk.Body()
    body.position = (x, y)
    shape = pymunk.Circle(body, r)
    shape.density = 3
    space.add(body, shape)
    return shape


class Ship:
    def __init__(self, space, v, x, y, theta):
        self.vertices = [(0,2), (0.5,1), (0.5,-1), (-0.5,-1), (-0.5,1)]
        #[(0,3), (1,1), (1,-2), (-1,-2), (-1,1)]
        self.body = pymunk.Body(1, 100, body_type=pymunk.Body.KINEMATIC)
        self.body.position = (x, y)
        self.body.velocity = v
        self.body.angle = math.radians(theta)
        self.shape = pymunk.Poly(self.body, self.vertices)

        #pymunk.Circle(self.body, r)
        space.add(self.body, self.shape)
        self.path_pos = 0

    def set_path_pos(self, path_pos):
        self.path_pos = path_pos


def main():
    n = 71
    r = 7  # radius of circular turns
    d = 1  # how far robot goes in straight line
    theta = 0  # measured clockwise from y axis (up)
    turning_radius = 1.99
    scale = 0.5
    start_pos = (math.floor(n / 2), 10, theta)
    # start_pos = (65,0,0)
    # goal_pos = (60, 65, 0)
    goal_pos = (35,60,0)
    # goal_pos = (20,50,90)
    cost_map = np.zeros((n, n))
    # list_of_obstacles = np.array([[12, 12, 10], [25, 25, 8], [38, 36, 4], [25, 55, 15]])
    # list_of_obstacles = np.array([[35,35,15]])
    # list_of_obstacles = np.array([[5,5,5], [20,30,5], [50,40,5], [10,60,5], [40,10,5], [30,30,5],[30,50,5]])
    list_of_obstacles = np.array([[30,5,5], [30,16,5],[30,27,5],[30,38,5],[30,49,5],[30,60,5],[30,68,2]])
    for row in list_of_obstacles:
        cost_map = cm.create_circle(row, cost_map, scale)

    # y is pointing up, x is pointing to the right
    edge_set_cardinal = [(0, 1, 0), (-1, 2, 45), (-2, 2, 90), (1, 2, 315), (2, 2, 270)]
    edge_set_ordinal = [(-1, 1, 0), (-2, 1, 45), (-3, 0, 90), (-1, 2, 315), (0, 3, 270)]

    ordinal_swaths = get_ordinal_swaths()

    # start_pos_test = (math.floor(n / 2), math.floor(n / 2))

    worked, L, edge_path, nodes_visited = a_star(start_pos, goal_pos, turning_radius, n, cost_map, edge_set_cardinal, edge_set_ordinal, ordinal_swaths)

    fig1, ax1 = plt.subplots(2,1)

    if worked:
        ax1[0].imshow(cost_map, origin='lower')
        xmax = 0
        ymax = 0
        PATH = [i for i in edge_path[::-1]]
        path = np.zeros((2,1))
        for i in range(np.shape(PATH)[0] - 1):
            P1 = PATH[i]
            P2 = PATH[i + 1]
            dubins_path = dubins.shortest_path((P1[0], P1[1], math.radians(P1[2] + 90) % (2*math.pi)), (P2[0], P2[1], math.radians(P2[2] + 90) % (2*math.pi)), turning_radius)
            configurations, _ = dubins_path.sample_many(0.2)
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
            ax1[0].plot(x, y, 'g')
            path = np.append(path, np.array([np.asarray(x).T, np.asarray(y).T]), axis=1)

        path = np.delete(path, 0, 1)
        print(np.shape(path))
        #np.savetxt("nodes.csv", path, delimiter=",")
        for obs in list_of_obstacles:
            ax1[0].add_patch(patches.Circle((obs[1],obs[0]),obs[2],fill=False))
    else:
        path = 0

    node_plot = np.zeros((n,n))
    for node in nodes_visited:
        node_plot[node[1], node[0]] = node_plot[node[1], node[0]] + 1
    ax1[1].imshow(node_plot, origin='lower')
    #plt.show()
    print("Total Cost:", L, sep=" ")



    space = pymunk.Space()
    space.gravity = (0, 0)
    initial_vel = Vec2d(0, 0)

    circles = []
    patch_list = []

    ship = Ship(space, initial_vel, start_pos[0], start_pos[1], start_pos[2])
    i = 0
    vs = np.zeros((5,2))
    for v in ship.shape.get_vertices():
        x, y = v.rotated(ship.body.angle) + ship.body.position
        vs[i][0] = x
        vs[i][1] = y
        i += 1

    ship_patch = patches.Polygon(vs, True)

    for row in list_of_obstacles:
        #circles.append(create_circle(space, row[0], row[1], row[2]))
        #patch_list.append(patches.Circle((row[0], row[1]), row[2], fill=False))
        circles.append(create_circle(space, row[1], row[0], row[2]))
        patch_list.append(patches.Circle((row[1], row[0]), row[2], fill=False))

    path = path.T
    heading_list = np.zeros(np.shape(path)[0])
    vel_path = np.zeros((np.shape(path)[0] - 1, np.shape(path)[1]))
    angular_vel = np.zeros(np.shape(vel_path)[0])

    for i in range(np.shape(vel_path)[0]):
        point1 = path[i,:]
        point2 = path[i + 1,:]
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
            heading = (math.atan2(velocity[1], velocity[0]) - math.pi/2 + 2 * math.pi) % (2 * math.pi)
        heading_list[i] = heading
        #print("velocity: ", velocity, sep=" ")
        vel_path[i,:] = velocity.T * 50
        #print(tuple(vel_path[i,:]))

    # set initial heading and final heading
    heading_list[0] = 0
    heading_list[-1] = 0

    # Estimate angular velocity at each point from current and next heading
    for i in range(np.shape(angular_vel)[0]):
        raw = heading_list[i+1] - heading_list[i]
        turn = min((-abs(raw)) % (2 * math.pi),abs(raw) % (2 * math.pi))
        if raw == 0:
            direction = -1
        else:
            direction = -abs(raw)/raw
        angular_vel[i] = direction*turn*30

    fig2 = plt.figure()
    ax2 = plt.axes(xlim=(0, n), ylim=(0, n))
    ax2.set_aspect("equal")

    def init():
        ax2.add_patch(ship_patch)
        for circle, patch in zip(circles, patch_list):
            ax2.add_patch(patch)
        return []


    def animate(dt, ship1, circles1, patch_list1):
        print(dt)
        # 20 ms step size
        for x in range(10):
            space.step(2 / 100 / 10)

        ship_pos = (ship.body.position.x, ship.body.position.y)
        print("path_node:",ship.path_pos,sep=" ")
        print("ship pos:",ship_pos,sep=" ")
        print("path pos:",path[ship.path_pos,:],sep=" ")
        if ship.path_pos < np.shape(vel_path)[0]:
            ship.body.velocity = Vec2d(vel_path[ship.path_pos, 0], vel_path[ship.path_pos, 1])
            ship.body.angular_velocity = angular_vel[ship.path_pos]
            if dist(ship_pos, path[ship.path_pos,:]) < 0.01:
                ship.set_path_pos(ship.path_pos + 1)

        animate_ship(dt, ship1)
        for circle, patch in zip(circles1, patch_list1):
            animate_obstacle(dt, circle, patch)
        return []


    def animate_ship(dt, patch):
        vs = np.asarray(ship.shape.get_vertices())
        heading = ship.body.angle
        R = np.asarray([[math.cos(heading), -math.sin(heading)],[math.sin(heading), math.cos(heading)]])
        vs = vs @ R + np.asarray(ship.body.position)
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
                     fargs=(ship_patch, circles, patch_list,),
                     interval=20,
                     blit=True,
                     repeat=False)

    plt.show()

if __name__ == "__main__":
    main()
