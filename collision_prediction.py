import math
from typing import List, Tuple

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d


def find_collision_point(swath: np.ndarray, x_pos, y_pos, radius) -> Tuple[float, float]:
    '''
    We check to see if a swath point is within the circle, starting from the beginning to the end
    When a swath point is found to be within the circle, the closest point on the circle to that point is taken as the
    collision point.
    '''
    print(x_pos, y_pos)
    print(radius)
    x = 0
    y = 0
    for coords in np.transpose((swath > 0).nonzero()):
        print(coords)
        x_diff = coords[1] - x_pos
        y_diff = coords[0] - y_pos
        print(x_diff,y_diff)
        print(math.sqrt(x_diff ** 2 + y_diff ** 2))
        if math.sqrt(x_diff ** 2 + y_diff ** 2) < radius:
            # find line from swath to center
            print("hi")
            theta = math.atan2(x_diff, y_diff)
            y = radius * math.cos(theta) + y_pos
            x = radius * math.sin(theta) + x_pos
            break

    return (x,y)


def find_normal(x_pos, y_pos, x_obs, y_obs) -> Vec2d:
    return Vec2d(x_pos - x_obs, y_pos - y_obs)


def collision(o1: pymunk.shapes.Shape, radius1: float, o2: pymunk.shapes.Shape, radius2: float, collision_point: Vec2d, e: float, n: Vec2d):
    m1 = o1.body.mass
    v1 = o1.body.velocity
    I1 = 1 / 2 * m1 * radius1 ** 2
    w1 = o1.body.angular_velocity
    r1 = o1.body.position - collision_point
    vp1 = v1 + w1.cross(r1)

    m2 = o2.body.mass
    v2 = o2.body.velocity
    I2 = 1 / 2 * m2 * radius2 ** 2
    w2 = o2.body.angular_velocity
    r2 = o2.body.position - collision_point
    vp2 = v2 + w2.cross(r2)

    vr = vp2 - vp1

    jr = - ((1 + e) * vr).dot(n)/(1/m1 + 1/m2 + (1/I1 * r1.cross(n).cross(r1) + 1/I2 * r2.cross(n).cross(r2)).dot(n))
    jr_hat = jr * n

    v1_new = v1 + jr_hat / m1
    v2_new = v2 + jr_hat / m2

    w1_new = w1 - jr * 1 / I1 * r1.cross(n)
    w2_new = w2 + jr * 1 / I2 * r2.cross(n)

    return v1_new, v2_new, w1_new, w2_new

