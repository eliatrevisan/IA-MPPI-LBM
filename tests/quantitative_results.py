import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')


home = os.path.expanduser("~")
directory = "/I-LSTM/src/"
model = "VGDNN"
file = "_summary.csv"

def main():
    results = getResults()

    fig, ax =  plt.subplots()
    ax.set_axis_off()
    printResults(ax, results)

    #print(results)
    fig.tight_layout()

    plt.show()

def getResults():
    """
    Return the data contained in csv file
    """
    #data = np.genfromtxt(home + directory + model + file, delimiter=",", skip_header=1, usecols=(0,2,3,4,5,6,7,8,9,10), dtype='unicode')
    data = np.genfromtxt(home + directory + model + file, delimiter=",", dtype='unicode')
    print(data)
    #for row in range(data.shape[0]):
        #for col in range(data.shape[1]);
            #data[row, col]
    return data

def printResults(ax, results):

    labels = results[0,:]
    print(labels)
    table = ax.table(cellText=results[1:,1:], colLabels=labels, loc='center')
    #table.set_fontsize(40)
    #table.scale(1.5, 1.5)

    return


if __name__ == "__main__":
    main()
