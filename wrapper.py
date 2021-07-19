from state_lattice import state_lattice_planner
import math
import numpy as np
import time

n = 300  # channel height
m = 40  # channel width
load_costmap_file = "sample_costmaps/test3.pk"  # "sample_costmaps/random_obstacles_1.pk"

# --- ship --- #
start_pos = (20, 10, 0)  # (x, y, theta)
goal_pos = (20, 282, 0)
initial_heading = math.pi / 2
turning_radius = 8
vel = 10  # constant linear velocity of ship
padding = 0  # padding around ship vertices to increase footprint when computing path costs

# --- primitives --- #
num_headings = 16

# --- ice --- #
num_obs = 100  # number of random ice obstacles
min_r = 1  # min ice radius
max_r = 5
upper_offset = 20  # offset from top of costmap where ice stops
lower_offset = 20  # offset from bottom of costmap where ice stops
allow_overlap = False  # if True allow overlap in ice obstacles
obstacle_density = 6
obstacle_penalty = 1

# --- A* --- #
g_weight = 0.3  # cost = g_weight * g_score + h_weight * h_score
h_weight = 0.7

# --- pid --- #
Kp = 3
Ki = 0.08
Kd = 0.5

# -- misc --- #
smooth_path = False  # if True run smoothing algorithm
replan = False # if True rerun A* search at each time step
save_animation = False  # if True save animation and don't show it
save_costmap = False

# weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
list_of_dist = []
list_of_times = []
for i in range(10):
    dist_moved, plan_time = state_lattice_planner(g_weight=g_weight, h_weight=h_weight, costmap_file=load_costmap_file,
                          start_pos=start_pos, goal_pos=goal_pos, initial_heading=initial_heading, padding=padding,
                          turning_radius=turning_radius, vel=vel, num_headings=num_headings,
                          num_obs=num_obs, min_r=min_r, max_r=max_r, upper_offset=upper_offset,
                          lower_offset=lower_offset, allow_overlap=allow_overlap, obstacle_density=obstacle_density,
                          obstacle_penalty=obstacle_penalty, Kp=Kp, Ki=Ki, Kd=Kd,
                          save_animation=save_animation, smooth_path=smooth_path, replan=replan, save_costmap=save_costmap)
    list_of_dist.append(dist_moved)
    list_of_times.append(plan_time)

string = ""
string1 = ""
with open('distance_test.csv', 'a') as f:
    for dist in list_of_dist:
        string = string + str(dist) + ","
    print(string, file=f)
    print(np.sum(list_of_dist)/10, file=f)
    for time in list_of_times:
        string1 = string1 + str(time) + ","
    print(string1, file=f)
    print(np.sum(list_of_times)/10, file=f)


#for i in range(10):
#    file_name = "Test_" + str(i)
#    state_lattice_planner(file_name=file_name, g_weight=0.2, h_weight=0.8, costmap_file="sample_costmaps/test2.pk", save_animation=False, smooth_path=True)
