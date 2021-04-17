import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches


def main():
    n = 21
    r = 2  # radius of circular turns
    d = 7  # how far robot goes in straight line
    theta = 0  # measured clockwise from y axis (up)
    grid = np.zeros((n, n))
    fig, ax = plt.subplots(1)
    # grid, arc_list = go_straight(math.floor(n / 2), math.floor(n/2), grid, d, 0)
    # grid, arc_list, goal_pos, valid = left_turn(math.floor(n / 2), math.floor(n / 2), grid, r, 45)
    # grid, arc_list, goal_pos, valid = right_turn(math.floor(n/2), math.floor(n/2), grid, r, 270)
    # grid, arc_list, goal_pos, valid = left_s_turn(math.floor(n / 2), math.floor(n / 2), grid, r, 315)
    # grid, arc_list, goal_pos, valid = right_s_turn(math.floor(n / 2), math.floor(n / 2), grid, r, 315)
    grid, arc_list, goal_pos, valid = left_turn_45(math.floor(n / 2), math.floor(n / 2), grid, r, 225)
    # grid, arc_list, goal_pos, valid = right_turn_45(math.floor(n / 2), math.floor(n / 2), grid, r, 225)
    ax.imshow(grid)
    if isinstance(arc_list, list):
        for arc in arc_list:
            ax.add_patch(arc)
    else:
        ax.add_patch(arc_list)

    ax.set_xticks(np.arange(-0.5, n, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    # ax.set_xticklabels(np.arange(0, n + 1, 1))
    # ax.set_yticklabels(np.arange(0, n + 1, 1))
    ax.grid(color='w', linestyle='-')
    plt.show()


def rot_matrix(theta):
    R = np.array([[math.cos(math.radians(theta)), -math.sin(math.radians(theta))],
                  [math.sin(math.radians(theta)), math.cos(math.radians(theta))]])
    return R


# fill in squares for circular arcs
def fill_squares(x_pos, y_pos, grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, dir):
    # left direction is positive, right direction is negative
    current_cell = np.array([x_pos, y_pos]).flatten()
    print(current_cell)
    print(x_center, y_center, sep= ",")
    adjacent_cell = np.zeros(2)
    loop = 1
    # valid becomes 1 when algorithm has reached goal without going out of bounds
    valid = 0
    while loop:
        for i in np.arange(0, 2, 1):
            if theta == 90:
                if i == 0:
                    adjacent_cell = current_cell + np.array([1, 0])
                elif i == 1:
                    adjacent_cell = current_cell + np.array([0, dir * -1])
                else:
                    adjacent_cell = current_cell + np.array([1, dir * -1])
            elif theta == 0:
                if i == 0:
                    adjacent_cell = current_cell + np.array([0, -1])
                elif i == 1:
                    adjacent_cell = current_cell + np.array([dir * -1, 0])
                else:
                    adjacent_cell = current_cell + np.array([dir * -1, -1])
            elif theta == 270:
                if i == 0:
                    adjacent_cell = current_cell + np.array([-1, 0])
                elif i == 1:
                    adjacent_cell = current_cell + np.array([0, dir * 1])
                else:
                    adjacent_cell = current_cell + np.array([-1, dir * 1])
            elif theta == 180:
                if i == 0:
                    adjacent_cell = current_cell + np.array([0, 1])
                elif i == 1:
                    adjacent_cell = current_cell + np.array([dir * 1, 0])
                else:
                    adjacent_cell = current_cell + np.array([dir * 1, 1])
            elif theta == 315:
                if dir == 1:
                    if current_cell[0] >= x_center_round:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, -1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, 1])
                else:
                    if current_cell[1] >= y_center:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, -1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, -1])
            elif theta == 45:
                if dir == 1:
                    if current_cell[1] >= y_center:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, -1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, -1])
                else:
                    if current_cell[0] <= x_center_round:
                        print("hi1")
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, -1])
                    else:
                        print("hi2")
                        if i == 0:
                            adjacent_cell = current_cell + np.array([1, 0])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([0, 1])
                        else:
                            adjacent_cell = current_cell + np.array([1, 1])
            elif theta == 135:
                if dir == 1:
                    if current_cell[0] <= x_center_round:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, 1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, -1])
                else:
                    if current_cell[1] <= y_center_round:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, 1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, 1])
            elif theta == 225:
                if dir == 1:
                    if current_cell[1] <= y_center_round:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, 1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([1, 1])
                else:
                    if current_cell[0] >= x_center_round:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, 1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, 1])
                    else:
                        if i == 0:
                            adjacent_cell = current_cell + np.array([0, -1])
                        elif i == 1:
                            adjacent_cell = current_cell + np.array([-1, 0])
                        else:
                            adjacent_cell = current_cell + np.array([-1, -1])

            print(adjacent_cell)
            v0 = np.sqrt(
                (adjacent_cell[0] - 0.5 - x_center) ** 2 + (adjacent_cell[1] - 0.5 - y_center) ** 2)
            v1 = np.sqrt(
                (adjacent_cell[0] + 0.5 - x_center) ** 2 + (adjacent_cell[1] - 0.5 - y_center) ** 2)
            v2 = np.sqrt(
                (adjacent_cell[0] + 0.5 - x_center) ** 2 + (adjacent_cell[1] + 0.5 - y_center) ** 2)
            v3 = np.sqrt(
                (adjacent_cell[0] - 0.5 - x_center) ** 2 + (adjacent_cell[1] + 0.5 - y_center) ** 2)
            print("v0", v0, "v1", v1, "v2", v2, "v3", v3, sep= " ")
            if (v0 > r > v2) or (v1 > r > v3) or (v2 > r > v0) or (v3 > r > v1):
                print("yes")
                if 0 <= adjacent_cell[0] < grid.shape[0] and 0 <= adjacent_cell[1] < grid.shape[0]:
                    grid[adjacent_cell[1], adjacent_cell[0]] = 1
                else:
                    loop = 0

                if current_cell[0] > adjacent_cell[0]:
                    dx = dx + 1
                elif current_cell[0] < adjacent_cell[0]:
                    dx = dx - 1
                elif current_cell[1] > adjacent_cell[1]:
                    dy = dy + 1
                elif current_cell[1] < adjacent_cell[1]:
                    dy = dy - 1

                # reached goal
                if dy == 0 and dx == 0:
                    loop = 0
                    valid = 1
                current_cell = adjacent_cell
                break
    return grid, valid


# create motion primitives
def go_straight(x_pos, y_pos, grid, d, theta):
    dx = round(d * math.sin(math.radians(theta)))
    dy = -round(d * math.cos(math.radians(theta)))
    goal_pos = np.array([x_pos + dx, y_pos + dy])

    if 0 < goal_pos[0] < grid.shape[0] and 0 < goal_pos[1] < grid.shape[0]:
        if abs(theta) == 45 or abs(theta) == 135:
            if dx < 0:
                arr = np.arange(x_pos + dx, x_pos + 1, 1)
                arr = np.flip(arr)
            else:
                arr = np.arange(x_pos, x_pos + dx + 1, 1)
            y = 0
            for x in arr:
                grid[y_pos + y, x] = 1
                if dy < 0:
                    y -= 1
                else:
                    y += 1
        else:
            if theta == 0:
                grid[y_pos - d:y_pos + 1, x_pos] = 1
            elif theta == 90:
                grid[y_pos, x_pos:x_pos + d + 1] = 1
            elif theta == -90:
                grid[y_pos, x_pos - d:x_pos + 1] = 1
            else:
                grid[y_pos:y_pos + d + 1, x_pos] = 1

    line = patches.FancyArrow(x_pos, y_pos, dx, dy, color='r')
    patches.ArrowStyle("-")
    return grid, line, goal_pos


def left_turn(x_pos, y_pos, grid, r, theta):
    valid = 0
    rotated = rot_matrix(theta) @ np.array([[-r, 0]]).T + np.array([[x_pos, y_pos]]).T
    rotated_round = np.rint(rotated).astype(int)
    x_center = rotated[0]
    y_center = rotated[1]
    x_center_round = rotated_round[0]
    y_center_round = rotated_round[1]

    goal_pos = np.rint(rot_matrix(theta) @ np.array([[-r, -r]]).T + np.array([[x_pos, y_pos]]).T).astype(int).flatten()
    dx = goal_pos[0] - x_pos
    dy = goal_pos[1] - y_pos

    # use actual x and y value of circle center to draw proper arc
    arc = patches.Arc((x_center, y_center), r * 2, r * 2, angle=theta, theta1=270, theta2=360, color='r')

    if 0 < goal_pos[0] < grid.shape[0] and 0 <= goal_pos[1] < grid.shape[0]:
        grid, valid = fill_squares(x_pos, y_pos, grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, 1)

    return grid, arc, goal_pos, valid


def right_turn(x_pos, y_pos, grid, r, theta):
    valid = 0
    rotated = rot_matrix(theta) @ np.array([[r, 0]]).T + np.array([[x_pos, y_pos]]).T
    rotated_round = np.rint(rotated).astype(int)
    x_center = rotated[0]
    y_center = rotated[1]
    x_center_round = rotated_round[0]
    y_center_round = rotated_round[1]

    goal_pos = np.rint(rot_matrix(theta) @ np.array([[r, -r]]).T + np.array([[x_pos, y_pos]]).T).astype(int).flatten()
    dx = goal_pos[0] - x_pos
    dy = goal_pos[1] - y_pos

    # use actual x and y value of circle center to draw proper arc
    arc = patches.Arc((x_center, y_center), r * 2, r * 2, angle=theta, theta1=180, theta2=270, color='r')

    if 0 < goal_pos[0] < grid.shape[0] and 0 < goal_pos[1] < grid.shape[0]:
        grid, valid = fill_squares(x_pos, y_pos, grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, -1)
    # use rounded value of x and y to nearest square when filling in cost map

    return grid, arc, goal_pos, valid


def left_s_turn(x_pos, y_pos, grid, r, theta):
    # do left turn
    arcs = []
    grid, arc1, new_pos, valid = left_turn(x_pos, y_pos, grid, r, theta)
    arcs.append(arc1)

    # do right turn
    valid = 0
    rotated = rot_matrix(theta) @ np.array([[-r, -2 * r]]).T + np.array([[x_pos, y_pos]]).T
    rotated_round = np.rint(rotated).astype(int)
    x_center = rotated[0]
    y_center = rotated[1]
    x_center_round = rotated_round[0]
    y_center_round = rotated_round[1]

    # new_pos = np.rint(rot_matrix(theta) @ np.array([[-r, -r]]).T + np.array([[x_pos, y_pos]]).T).astype(int)
    goal_pos = np.rint(rot_matrix(theta) @ np.array([[-2 * r, -2 * r]]).T + np.array([[x_pos, y_pos]]).T).astype(int).flatten()
    dx = goal_pos[0] - new_pos[0]
    dy = goal_pos[1] - new_pos[1]

    arcs.append(patches.Arc((x_center, y_center), r * 2, r * 2, angle=theta, theta1=90, theta2=180, color='r'))
    theta = (theta - 90 + 360) % 360  # make sure the angle is between 0 and 360

    if 0 < goal_pos[0] < grid.shape[0] and 0 < goal_pos[1] < grid.shape[0]:
        grid, valid = fill_squares(new_pos[0], new_pos[1], grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, -1)

    return grid, arcs, goal_pos, valid


def right_s_turn(x_pos, y_pos, grid, r, theta):
    # do right turn
    arcs = []
    grid, arc1, new_pos, valid = right_turn(x_pos, y_pos, grid, r, theta)
    arcs.append(arc1)

    # do left turn
    valid = 0
    rotated = rot_matrix(theta) @ np.array([[r, -2 * r]]).T + np.array([[x_pos, y_pos]]).T
    rotated_round = np.rint(rotated).astype(int)
    x_center = rotated[0]
    y_center = rotated[1]
    x_center_round = rotated_round[0]
    y_center_round = rotated_round[1]

    arcs.append(patches.Arc((x_center, y_center), r * 2, r * 2, angle=theta, theta1=0, theta2=90, color='r'))

    # new_pos = np.rint(rot_matrix(theta) @ np.array([[r, -r]]).T + np.array([[x_pos, y_pos]]).T).astype(int)
    goal_pos = np.rint(rot_matrix(theta) @ np.array([[2 * r, -2 * r]]).T + np.array([[x_pos, y_pos]]).T).astype(int).flatten()
    dx = goal_pos[0] - new_pos[0]
    dy = goal_pos[1] - new_pos[1]

    theta = (theta + 90) % 360  # make sure the angle is between 0 and 360

    if 0 < goal_pos[0] < grid.shape[0] and 0 < goal_pos[1] < grid.shape[0]:
        grid, valid = fill_squares(new_pos[0], new_pos[1], grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, 1)

    return grid, arcs, goal_pos, valid


def left_turn_45(x_pos, y_pos, grid, r, theta):
    valid = 0
    rotated = rot_matrix(theta) @ np.array([[-r, 0]]).T + np.array([[x_pos, y_pos]]).T
    rotated_round = np.rint(rotated).astype(int)
    x_center = rotated[0]
    y_center = rotated[1]
    x_center_round = rotated_round[0]
    y_center_round = rotated_round[1]

    arc = patches.Arc((x_center, y_center), r * 2, r * 2, angle=360-theta, theta1=315, theta2=360, color='r')
    goal_pos = np.rint((rot_matrix(theta) @ (rot_matrix(-45) @ np.array([[r, 0]]).T) + np.array([[x_center_round, y_center_round]]).T)).astype(int).flatten()
    dx = goal_pos[0] - x_pos
    dy = goal_pos[1] - y_pos

    # if 0 < goal_pos[0] < grid.shape[0] and 0 < goal_pos[1] < grid.shape[0]:
        # grid, valid = fill_squares(x_pos, y_pos, grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, 1)

    valid = 1
    return grid, arc, goal_pos, valid


def right_turn_45(x_pos, y_pos, grid, r, theta):
    valid = 0
    rotated = rot_matrix(theta) @ np.array([[r, 0]]).T + np.array([[x_pos, y_pos]]).T
    rotated_round = np.rint(rotated).astype(int)
    x_center = rotated[0]
    y_center = rotated[1]
    x_center_round = rotated_round[0]
    y_center_round = rotated_round[1]

    arc = patches.Arc((x_center, y_center), r * 2, r * 2, angle=theta, theta1=180, theta2=225, color='r')

    goal_pos = np.rint((rot_matrix(theta) @ (rot_matrix(45) @ np.array([[-r, 0]]).T) + np.array([[x_center_round, y_center_round]]).T)).astype(int).flatten()
    dx = goal_pos[0] - x_pos
    dy = goal_pos[1] - y_pos

    if 0 < goal_pos[0] < grid.shape[0] and 0 < goal_pos[1] < grid.shape[0]:
        grid, valid = fill_squares(x_pos, y_pos, grid, x_center, y_center, x_center_round, y_center_round, r, dx, dy, theta, -1)

    return grid, arc, goal_pos, valid


if __name__ == "__main__":
    main()
