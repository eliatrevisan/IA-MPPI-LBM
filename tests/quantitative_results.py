import sys
import os
import csv
import numpy as np
import pandas as pd
from io import StringIO
sys.path.append('../')


home = os.path.expanduser("~")
directory = "/I-LSTM/src/"
model = "VGDNN"
file = "_summary.csv"

def main():
    results = getResults()
    print(results)

def getResults():
    """
    Return the data contained in csv file
    """
    #data = np.genfromtxt(home + directory + model + file, delimiter=",", skip_header=1, usecols=(0,2,3,4,5,6,7,8,9,10), dtype='unicode')
    data = np.genfromtxt(home + directory + model + file, delimiter=",", dtype='unicode')
    return data

def printResults(results):



if __name__ == "__main__":
    main()
