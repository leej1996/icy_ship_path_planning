import math
from typing import Tuple

import numpy as np
import pymunk
from pymunk import Vec2d


class Ship:
    def __init__(self, vertices: np.ndarray, start_pos: Tuple, initial_heading: float, turning_radius: float):
        self.vertices = vertices
        self.initial_heading = initial_heading  # angle between ship and fixed coordinates on the map
        self.turning_radius = turning_radius
        dist = lambda a, b: np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        self.max_ship_length = np.ceil(max(dist(a, b) for a in vertices for b in vertices)).astype(int)
        assert self.max_ship_length != 0, 'ship length cannot be 0'

        # setup for pymunk
        self.body = pymunk.Body(1, 100, body_type=pymunk.Body.KINEMATIC)
        self.body.position = start_pos[:2]
        self.body.velocity = Vec2d(0, 0)
        self.body.angle = 0
        self.shape = pymunk.Poly(self.body, [tuple(item) for item in self.vertices])  # uses same ship vertices
        self.path_pos = 0

    def set_path_pos(self, path_pos):
        self.path_pos = path_pos

    @staticmethod
    def calc_turn_radius(rate, speed):
        """
        rate: deg/min
        speed: knots
        """
        theta = rate * math.pi / 180  # convert to rads
        s = speed * 30.8667  # convert to m
        return s / theta
