import numpy as np
import math
from matplotlib import pyplot as plt
import dubins

turning_radius = 0.99999

fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
ax.set_aspect("equal")

cardinal_swaths = dict()
array = np.zeros((11, 11))

edge_set_cardinal = [(1, 0, 0),
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

test_set = [(0, 4, 3)]
# Initial set at 0 degrees pointing up
R = np.asarray(
    [[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0], [math.sin(math.pi / 2), math.cos(math.pi / 2), 0], [0, 0, 1]])
start_pos = (5, 5, 1)
for e in edge_set_ordinal:
    heading = start_pos[2] * 45
    array = np.zeros((11, 11))
    swath = [[start_pos[0], start_pos[1]]]
    rotated_e = R @ np.asarray(e).T + np.array([start_pos[0], start_pos[1], 0])
    dubins_path = dubins.shortest_path((start_pos[0], start_pos[1], math.radians((start_pos[2] + 2) * 45)),
        (rotated_e[0], rotated_e[1], math.radians((rotated_e[2] + 2) * 45) % (2 * math.pi)), turning_radius)
    configurations, _ = dubins_path.sample_many(0.01)
    # 0.01
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
        array[pair[1], pair[0]] = 1
    cardinal_swaths[e, 0 + heading] = array
    cardinal_swaths[e, 90 + heading] = np.flip(array.T, 1)  # Rotate 90 degrees CCW
    cardinal_swaths[e, 180 + heading] = np.flip(np.flip(array, 1), 0)  # Rotate 180 degrees CCW
    cardinal_swaths[e, 270 + heading] = np.flip(array.T, 0)  # Rotate 270 degrees CCW
    ax.plot(x, y)

plt.imshow(cardinal_swaths[(0, 3, 2),90])

plt.show()
