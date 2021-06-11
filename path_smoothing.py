import math
import dubins
import numpy as np
from skimage import draw


def path_smoothing(path, path_length, cost_map, turning_radius,
                   start, goal, nodes, vertices, dist_cuttoff=100, eps=1e-4):  # epsilon handles small error from dubins package
    print("Attempt Smoothing")
    chan_h, chan_w = np.shape(cost_map)
    total_length = np.sum(path_length)
    # probability is based on length between nodes (greater length = greater probability)
    probabilities = np.asarray(path_length) / total_length
    x = []
    y = []

    for vi in path:
        x.append(vi[0])
        y.append(vi[1])

    # determine between which current nodes on path nodes will be added based off previous probabilities
    # generates a list where each value is an index corresponding to a segment between two nodes on the path
    segments = np.sort(np.random.choice(np.arange(len(path)), nodes, p=probabilities))

    added_x = []
    added_y = []
    counter = 0
    offset = 0

    while counter < len(segments):
        node_id = segments[counter]  # index where node will be added
        num_values = np.shape(segments[segments == node_id])[0]  # number of nodes to be added

        # two current path nodes where the new nodes will be added
        node = path[node_id + offset]
        prev_node = path[node_id + offset - 1]

        # sample points between nodes
        prim = dubins.shortest_path((prev_node[0], prev_node[1], math.radians((prev_node[2] + 2) * 45)),
                                    (node[0], node[1], math.radians((node[2] + 2) * 45)), turning_radius)
        configurations, _ = prim.sample_many(0.1)

        # if there are multiple nodes to be added between two nodes, try to space them out equally
        values = [configurations[int(i)] for i in
                  np.linspace(0, len(configurations), num=num_values + 2, endpoint=False)]
        values.pop(0)
        values.pop()

        # actually insert nodes into the path
        inc = 1
        for v in values:
            heading = v[2] - math.pi / 2
            if 0 <= v[0] < chan_w and 0 <= v[1] < chan_h:
                added_x.append(v[0])
                added_y.append(v[1])
                if heading < 0:
                    heading = heading + 2 * math.pi

                path.insert(node_id - 1 + inc + counter, (v[0], v[1], heading / (math.pi / 4)))
                inc += 1
        counter += num_values
        offset = offset + inc - 1

    # initialize smoothing algorithm
    prev = {}
    smooth_cost = {}
    for i, vi in enumerate(path):
        smooth_cost[vi] = np.inf
        prev[vi] = None
    smooth_cost[path[0]] = 0

    for i, vi in enumerate(path):
        for j, vj in enumerate(path[i + 1:]):
            # determine cost between node vi and vj
            invalid = False
            dubins_path = dubins.shortest_path((vi[0], vi[1], math.radians((vi[2] + 2) * 45) % (2 * math.pi)),
                                               (vj[0], vj[1], math.radians((vj[2] + 2) * 45) % (2 * math.pi)),
                                               turning_radius - eps)

            if dubins_path.path_length() > dist_cuttoff:
                if j != 0:
                    break

            configurations, _ = dubins_path.sample_many(1.2)
            swath = np.zeros_like(cost_map, dtype=bool)

            # for each point sampled on dubins path, get x, y, theta
            for config in configurations:
                x_cell = int(round(config[0]))
                y_cell = int(round(config[1]))

                theta = config[2] - math.pi / 2
                R = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

                # rotate/translate vertices of ship from origin to sampled point with heading = theta
                rot_vi = np.round(np.array([[x_cell], [y_cell]]) + R @ vertices.T).astype(int)

                # check if any vertex of ship is outside of cost map (invalid path)
                for v in rot_vi.T:
                    if not (0 <= v[0] < chan_w and 0 <= v[1] < chan_h):
                        invalid = True

                if invalid:
                    break

                # draw rotated ship polygon and put occupied cells into a mask
                rr, cc = draw.polygon(rot_vi[1, :], rot_vi[0, :])
                swath[rr, cc] = True

            if invalid:
                continue

            # determine smooth cost and compare to see if it is cheaper than cost from a different node
            swath_cost = np.sum(cost_map[swath])
            adj_cost = float(swath_cost + dubins_path.path_length())
            if smooth_cost[vi] + adj_cost < smooth_cost[vj]:
                smooth_cost[vj] = smooth_cost[vi] + adj_cost
                prev[vj] = vi

    # reconstruct path
    smooth_path = [goal]
    node = goal
    while node != start:
        prior_node, node = node, prev[node]
        assert prior_node[1] >= node[1], "sequential nodes should always move forward in the y direction"
        smooth_path.append(node)

    return smooth_path, x, y, added_x, added_y
