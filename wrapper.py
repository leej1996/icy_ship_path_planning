from state_lattice import state_lattice_planner

'''
weight_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in weight_list:
    g_weight = i
    h_weight = 1 - g_weight
    print("a= ", g_weight, " b= ", h_weight)
    state_lattice_planner(g_weight=g_weight, h_weight=h_weight, costmap_file="sample_costmaps/test1.pk", save_animation=False)
'''

for i in range(10):
    file_name = "Test_" + str(i)
    state_lattice_planner(file_name=file_name, g_weight=0.2, h_weight=0.8, costmap_file="sample_costmaps/test2.pk", save_animation=False, smooth_path=True)
