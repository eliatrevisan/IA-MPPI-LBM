import os
import csv
import numpy as np


home = os.path.expanduser("~")
directory = "/data/"

def main():
    
    names = getFileNames()
    data = []
    agents = 0
    two = 0
    three = 0
    timestamp = 0

    for file in names:

        content = getDataFromFile(file)
        n_agents = getNumofAgents(content)

        #TODO Make a function to discard the first and last 10 time steps (first and last second). Or different amount of time in better 

        if n_agents == 2:
            two += 1
        elif n_agents == 3:
            three += 1

        content = reindex_agents(content, agents)
        content = reindex_timestamp(content, timestamp)
        agents = agents + n_agents
        timestamp = last_timestep(content)
        data.append(content)
    
    save_to_file(data)

def getDataFromFile(file):
    """
    Return the data contained in csv file
    """
    data = np.genfromtxt(home + directory + file, delimiter=",")[11:-10,:]
    return data

def getNumofAgents(file):
    """
    Return the number of agents in data file
    """
    return len(np.unique(file[:,0]))

def save_to_file(data):
    """
    Save data to csv file
    """
    fields = ['id', 'timestep_s', 'timestep_ns', 'pos_x', 'pox_y','vel_x', 'vel_y', 'goal_x', 'goal_y' ]

    with open(home + directory + 'datafile.csv', 'w+') as f:
        write = csv.writer(f)
        write.writerow(fields)
        for i in range(len(data)):
            write.writerows(data[i])


def reindex_agents(data, agents):
    """
    Reindex the roboat ID's
    """
    for i in range(len(data)):
        if data[i][0] == 0:
            data[i][0] = agents + 1
        elif data[i][0] == 1:
            data[i][0] = agents + 2
        elif data[i][0] == 2:
            data[i][0] = agents + 3
    return data

def reindex_timestamp(data, timestamp):
    """
    Reindex the timestamp to make final file continuous in time
    """
    dt = 0.1
    data[:,1] = data[:,1] + timestamp + dt
    return data

def last_timestep(data):
    last = len(data) - 1
    return data[last][1]

def getFileNames():
    """
    Retrieve file names of all csv files in specified directory
    """
    names = []
    for root, dirs, files in os.walk(home + directory):
        for file in files:
            if file.endswith('.csv'):
                names.append(file)

    return names


if __name__ == "__main__":
    main()