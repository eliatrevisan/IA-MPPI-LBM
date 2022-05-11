import scipy.io
import sys
sys.path.append('../')
import os
import matplotlib.pyplot as plt
from src.data_utils.plot_utils import *
import src.data_utils.Support as sup
from src.data_utils.Recorder import Recorder as rec

home = os.path.expanduser("~")
filepath = "/I-LSTM/trained_models/VGDNN/"
exp_num = "1/"
name = "roboat_results.mat"

class Map:
    size = [1715, 881]
    data = plt.imread(home + '/I-LSTM/data/simulation/roboat/map.png')
    origin = [-78, -40, 0.0]
    resolution = 0.081

def plot_map(map, ax):
    ax.imshow(map.data,
               extent = (map.origin[0], map.origin[0] + map.size[0] * map.resolution,
                         map.origin[1], map.origin[1] + map.size[1] * map.resolution),
               cmap='gray_r')
    ax.set_facecolor('#262626')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #ax.yaxis.set_label_coords(-0.08, .5)
    #ax.xaxis.set_label_coords(0.5, -0.09)

mat = scipy.io.loadmat(home + filepath + exp_num + name)
"""
print(mat['predictions'][0][0][0])
print("2")
print(mat['predictions'][0][0][1])
print("2")
print(mat['predictions'][0][0][2])
print("2")
print(len(mat['predictions'][0][0]))

"""

print(mat['__version__'])



#sup.path_from_vel(initial_pos=np.array([0,0]), pred_vel=mat['predictions'][0][0]  , dt=0.4)

#sup.path_from_vel(initial_pos=np.array([0,0]),
#    pred_vel=pred_vel_global_frame, dt=self.args.dt)

#rec.animate_global(input_list, grid_list, all_predictions, y_ground_truth_list,
#                       other_agents_list,
#                       trajectories, all_traj_likelihood,test_args)

map = Map()

fig, ax = plt.subplots()
ax.plot(mat['trajectories'][0][0][0][0][1][:,0], mat['trajectories'][0][0][0][0][1][:,1])
plot_map(map, ax)
#plt.show()
