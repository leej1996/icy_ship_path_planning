import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import patches
import pymunk
from pymunk.vec2d import Vec2d

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
patch = plt.Circle((5, -5), 0.75, fc='y')

velocity = (0,10)
w, h = 10, 20
vs = np.array([[0,2], [0.5,1], [0.5,-1], [-0.5,-1], [-0.5,1]])
vs_pymunk = [(0,2), (0.5,1), (0.5,-1), (-0.5,-1), (-0.5,1)]
#print(np.asarray(vs_pymunk))
b = pymunk.Body()
b.position = 1, 1
b.velocity = Vec2d(0,10)
poly_good = pymunk.Poly(b, vs_pymunk)

i = 0
print(np.asarray(poly_good.get_vertices()))
for v in poly_good.get_vertices():
    x, y = v.rotated(poly_good.body.angle) + poly_good.body.position
    vs[i][0] = x
    vs[i][1] = y
    print(x, y)
    i += 1

vs = np.asarray(poly_good.get_vertices())
heading = poly_good.body.angle
R = np.asarray([[math.cos(heading), -math.sin(heading)],[math.sin(heading), math.cos(heading)]])
vs = vs @ R + np.asarray(poly_good.body.position)

print(vs)
poly_patch = plt.Polygon(vs, True)
'''
def init():
    ax.add_patch(poly_patch)
    return patch,

def animate(i):
    x, y = patch.center
    x = 5 + 3 * np.sin(np.radians(i))
    y = 5 + 3 * np.cos(np.radians(i))
    
    poly_patch.set_xy()
    patch.center = (x, y)
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=20,
                               blit=True)
'''
ax.add_patch(poly_patch)

plt.show()
