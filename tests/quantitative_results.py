import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../')


home = os.path.expanduser("~")
directory = "/I-LSTM/src/"
model = "VGDNN"
filename = "_summary.csv"

def main():
    models, labels, results = getResults()

    fig, ax =  plt.subplots(2)
    printResults(ax, models, results,labels)

    #print(results)
    #fig.tight_layout()

    plt.show()

def getResults():
    """
    Return the data contained in csv file
    """
    data = np.genfromtxt(home + directory + model + filename, delimiter=",", skip_header=1, usecols=(2,3,4,5,6,7,8,9,10))
    labels = np.genfromtxt(home + directory + model + filename, delimiter=",", dtype=str)[0,:]
    models = np.genfromtxt(home + directory + model + filename, delimiter=",", usecols=0, skip_header=1, dtype=str)

    data[:,2] 

    for i in range(len(data[:,2])):
        data[i,2] = np.round(data[i,2], 2)
        data[i,3] = np.round(data[i,3], 2)
        data[i,4] = np.round(data[i,4], 2)

    return models, labels, data

def printResults(ax, models, results,labels):

    MSE = results[:,2]
    FDE = results[:,3]

    N = len(MSE)
    ind = np.arange(N)  
    barwidth = 0.25

    ax[0].bar(ind + barwidth * 0.5, MSE, align='center', width=barwidth, label='MSE')
    ax[0].bar(ind + barwidth *1.5, FDE, align='center', width=barwidth, label='FDE')

    for index, value in enumerate(MSE):
        ax[0].text(index , value + 0.1,
             str(round(value,2)), fontsize=8)
    for index, value in enumerate(FDE):
        ax[0].text(index + barwidth * 1.0, value + 0.1,
             str(round(value,2)), fontsize=8)


    ax[0].legend()
    ax[0].set_title('Mean Squared Error and Final Displacement Error')
    ax[0].set_xlabel('Models')
    ax[0].set_ylabel('Error')
    ax[0].set_xticks(ind+barwidth)
    ax[0].set_xticklabels( models )
    print(labels)

    table = list(results[:,[1,4,5,6,7,8]])#,3,4,5,6]])
    table = np.reshape(table, (-1,6)).transpose()
    labels = labels[[3,6,7,8,9,10]]
    
    print(labels)
    print(table)

    ax[1].table(cellText=table, rowLabels=labels, colLabels=models, loc='center')
    ax[1].axis('off')
    
    return


if __name__ == "__main__":
    main()
