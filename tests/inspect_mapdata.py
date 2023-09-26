import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('../')
import src.data_utils.Support as sup
import src.data_utils.AgentContainer as ag_cont
import cv2


home = os.path.expanduser("~")
datapath = "/I-LSTM/data/simulation/"
ped = "6_peds_corridor/"
roboat = "roboat/"
mapname = "map.png"


class Map:
    grid = plt.imread(home+datapath+roboat + 'map.png')
    size = [1715, 881]
    origin = [-78, -40]
    resolution = 0.081
    submap_width = 9.72*2
    submap_height = 9.72*2

def main():

    pedmap = plt.imread(home + datapath + ped + mapname)
    roboatmap = plt.imread(home + datapath + roboat + mapname)

    map = Map()
    agcont = ag_cont.AgentContainer()
    agent_pos = [1.0, 1.0]
    agcont.occupancy_grid.resolution = map.resolution
    agcont.occupancy_grid.gridmap = map.grid
    agcont.occupancy_grid.map_size = map.size

    name = home + datapath + ped + mapname

    grid = agcont.occupancy_grid.getSubmapByCoords(agent_pos[0], agent_pos[1], map.submap_width, map.submap_height)

    print(grid.shape[0], grid.shape[1])

    fig, ax = plt.subplots(1,3)
    sup.plot_grid_roboat(ax[0], map.origin, map.grid, map.resolution, map.size)
    ax[0].plot(agent_pos[0], agent_pos[1], marker='.', markersize=10)
    ax[1].imshow(grid)
    ax[2].imshow( cv2.resize(grid, (60,60)) )
    plt.show()

if __name__ == "__main__":
    main()





