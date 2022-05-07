import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append('../')
import src.data_utils.Support as sup


home = os.path.expanduser("~")
datapath = "/I-LSTM/data/simulation/"
ped = "6_peds_corridor/"
roboat = "roboat/"
mapname = "map.png"

class Map:
    size = [1715, 881]
    origin = [-78, -40]
    resolution = 0.081

def main():

    pedmap = plt.imread(home + datapath + ped + mapname)
    roboatmap = plt.imread(home + datapath + roboat + mapname)

    map = Map()

    print(pedmap.shape)
    print(roboatmap.shape)

    name = home + datapath + ped + mapname

    sup.create_map_from_png(name, map.resolution, map.size, map.origin, data_path=home+datapath+ped)

if __name__ == "__main__":
    main()





