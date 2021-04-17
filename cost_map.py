import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    n = 51
    scale = 1
    cost_map = np.ones((n, n))
    list_of_obstacles = np.array([[10, 10, 7], [25, 25, 10], [40,40,6]])
    for row in list_of_obstacles:
        cost_map = create_circle(row, cost_map, scale)
    plt.imshow(cost_map)
    plt.show()


def determine_cost(x, y, x_center, y_center, r, scale):
    # Find cost of going up from grid
    x = np.asarray(x)
    y = np.asarray(y)
    d = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    # cost will become higher closer into the circle the square is
    cost = ((r - d) ** 2 + 1)*scale
    return cost


def create_circle(data, cost_map, scale):
    # Use Midpoint Circle Drawing Algorithm
    x_center = data[0]
    y_center = data[1]
    r = data[2]
    x = r
    y = 0

    # Generate four points in cardinal directions first r distance away from center
    if r > 0:
        cost_map[r + x_center, y_center] = determine_cost(r + x_center, y_center, x_center, y_center, r, scale)
        cost_map[x_center, r + y_center] = determine_cost(x_center, r + y_center, x_center, y_center, r, scale)
        cost_map[x_center, -r + y_center] = determine_cost(x_center, -r + y_center, x_center, y_center, r, scale)
        cost_map[-r + x_center, y_center] = determine_cost(-r + x_center, y_center, x_center, y_center, r, scale)
        cost_map[-r + x_center:r + x_center + 1, y_center] = determine_cost(np.arange(-r + x_center, r + x_center + 1), y_center, x_center, y_center, r, scale)
    else:
        # When radius is 0, print a point
        cost_map[x_center, y_center] = 2

    # initialise value of P
    p = 1 - r

    while x > y:
        y += 1

        if p <= 0:
            p = p + 2 * y + 1
        else:
            x -= 1
            p = p + 2 * y - 2 * x + 1

        if x < y:
            break

        # Draw the circle by reflecting over the four quadrants
        cost_map[x + x_center, y + y_center] = determine_cost(x + x_center, y + y_center, x_center, y_center, r, scale)
        cost_map[-x + x_center, y + y_center] = determine_cost(-x + x_center, y + y_center, x_center, y_center, r, scale)
        cost_map[x + x_center, -y + y_center] = determine_cost(x + x_center, -y + y_center, x_center, y_center, r, scale)
        cost_map[-x + x_center, -y + y_center] = determine_cost(-x + x_center, -y + y_center, x_center, y_center, r, scale)
        cost_map[-x + x_center:x + x_center + 1, -y + y_center] = determine_cost(np.arange(-x + x_center, x + x_center + 1), -y + y_center, x_center, y_center, r, scale)
        cost_map[-x + x_center:x + x_center + 1, y + y_center] = determine_cost(np.arange(-x + x_center, x + x_center + 1), y + y_center, x_center, y_center, r, scale)

        if x != y:
            cost_map[y + x_center, x + y_center] = determine_cost(y + x_center, x + y_center, x_center, y_center, r, scale)
            cost_map[-y + x_center, x + y_center] = determine_cost(-y + x_center, x + y_center, x_center, y_center, r, scale)
            cost_map[y + x_center, -x + y_center] = determine_cost(y + x_center,-x + y_center, x_center, y_center, r, scale)
            cost_map[-y + x_center, -x + y_center] = determine_cost(-y + x_center, -x + y_center, x_center, y_center, r, scale)
            cost_map[-y + x_center:y + x_center + 1, x + y_center] = determine_cost(np.arange(-y + x_center, y + x_center + 1), x + y_center, x_center, y_center, r, scale)
            cost_map[-y + x_center:y + x_center + 1, -x + y_center] = determine_cost(np.arange(-y + x_center, y + x_center + 1), -x + y_center, x_center, y_center, r, scale)

    return cost_map


if __name__ == "__main__":
    main()
