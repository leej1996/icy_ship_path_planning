import pymunk
import numpy as np
import math
import pymunk.matplotlib_util
from pymunk.vec2d import Vec2d
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import patches
import cost_map as cm


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


def dist(u,v):
    """
    Returns euclidean distance between points u = (x1, y1), v = (x2, y2)
    """
    x1 = u[0]
    y1 = u[1]
    x2 = v[0]
    y2 = v[1]

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

'''

'''
n = 71
scale = 1
space = pymunk.Space()
space.gravity = (0, 0)
cost_map = np.zeros((n, n))
start_pos = (math.floor(n / 2), 0, 0)
v = Vec2d(0, 0)

circles = []
patch_list = []
#list_of_obstacles = np.array([[10, math.floor(n / 2), 5], [30, 10, 5], [30, math.floor(n / 2), 5]])
list_of_obstacles = np.array([[5,30,5], [16,30,5],[27,30,5],[38,30,5],[49,30,5],[60,30,5],[68,30,2]])

ship = Ship(space, v, start_pos[0], start_pos[1], 1)
i = 0
vs = np.zeros((5,2))
for v in ship.shape.get_vertices():
    x, y = v.rotated(ship.body.angle) + ship.body.position
    vs[i][0] = x
    vs[i][1] = y
    i += 1

#ship_patch = patches.Circle((ship.body.position.x, ship.body.position.y), 1, fill=False)
ship_patch = patches.Polygon(vs, True)

for row in list_of_obstacles:
    cost_map = cm.create_circle(row, cost_map, scale)
    circles.append(create_circle(space, row[0], row[1], row[2]))
    patch_list.append(patches.Circle((row[0], row[1]), row[2], fill=False))

path = np.genfromtxt('nodes.csv', delimiter=',').T
print(np.shape(path))
heading_list = np.zeros(np.shape(path)[0])

vel_path = np.zeros((np.shape(path)[0] - 1, np.shape(path)[1]))
angular_vel = np.zeros(np.shape(vel_path)[0])
print(np.shape(angular_vel))
print(np.shape(vel_path))

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
        slope = velocity[1]/velocity[0]
        heading = (math.atan2(velocity[1], velocity[0]) - math.pi/2 + 2 * math.pi) % (2 * math.pi)
    heading_list[i] = heading
    #print("velocity: ", velocity, sep=" ")
    vel_path[i,:] = velocity.T * 50
    #print(tuple(vel_path[i,:]))

# set initial heading and final heading
heading_list[0] = 0
heading_list[-1] = 0

for i in range(np.shape(angular_vel)[0]):
    raw = heading_list[i+1] - heading_list[i]
    turn = min((-abs(raw)) % (2 * math.pi),abs(raw) % (2 * math.pi))
    if raw == 0:
        direction = -1
    else:
        direction = -abs(raw)/raw
    angular_vel[i] = direction*turn*30

    print(i, angular_vel[i], sep=" ")
    print(heading_list[i+1], heading_list[i], sep=", ")
    #print(i)

print(tuple(vel_path[0,:]))
#ship.body.velocity = Vec2d(vel_path[0,0],vel_path[0,1])
fig = plt.figure()
ax = plt.axes(xlim=(0, n), ylim=(0, n))
ax.set_aspect("equal")

#vel_path = np.zeros_like(vel_path)
#'''

def init():
    ax.add_patch(ship_patch)
    for circle, patch in zip(circles, patch_list):
        ax.add_patch(patch)
    return []


def animate(dt, ship1, circles1, patch_list1):
    print(dt)
    for x in range(10):
        space.step(2 / 100 / 10)

    #print(vel_path[dt])
    ship_pos = (ship.body.position.x, ship.body.position.y)
    print("path_node:",ship.path_pos,sep=" ")
    print("ship pos:",ship_pos,sep=" ")
    print("path pos:",path[ship.path_pos,:],sep=" ")
    if ship.path_pos < np.shape(vel_path)[0]:
        ship.body.velocity = Vec2d(vel_path[ship.path_pos, 0], vel_path[ship.path_pos, 1])
        ship.body.angular_velocity = angular_vel[ship.path_pos]
        #ship.body.angular_velocity = 1
        if dist(ship_pos, path[ship.path_pos,:]) < 0.01:
            ship.set_path_pos(ship.path_pos + 1)

    animate_ship(dt, ship1)
    for circle, patch in zip(circles1, patch_list1):
        animate_obstacle(dt, circle, patch)
    return []


def animate_ship(dt, patch):
    #patch.center = (ship.body.position.x, ship.body.position.y)
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
anim = animation.FuncAnimation(fig,
                     animate,
                     init_func=init,
                     frames=frames,
                     fargs=(ship_patch, circles, patch_list,),
                     interval=20,
                     blit=True,
                     repeat=False)
#'''
plt.show()
'''
f = r"C://Users/Justin/Documents/EMARO+/Centrale Nantes/Courses/Thesis/Cost Map/animation.gif"
writergif = animation.PillowWriter(fps=30)
anim.save(f, writer=writergif)

'''
