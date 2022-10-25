
import os
import matplotlib.pyplot as plt
import numpy as np


home = os.path.expanduser("~")
#filepath = "/I-LSTM/data/roboat/"
filepath = "/roboat_data/"

def print_results(scenario):
    data = getDataFromFile(scenario)
    print("Scenario: ", scenario)
    # Total number of identical agents
    print(data.shape)
    print(data[0,:])

    print("Unique Agents: ", len(np.unique(data[:,0])))

    print("Unique time steps: ", len(np.unique(data[:,1])))

def getDataFromFile(scenario):
    """
    Return the data contained in csv file
    """
    data = np.genfromtxt(home + filepath + scenario + "/total_log_open_crossing.csv", delimiter=",")[11:-10,:]
    return data

def main():
    #scenario = "herengracht"
    #print_results(scenario)
    #scenario = "prinsengracht"
    #print_results(scenario)
    #scenario = "bloemgracht"
    #print_results(scenario)
    scenario = "all_data"
    print_results(scenario)



if __name__ == "__main__":
    main()