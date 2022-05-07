"""
New independent Keras DataHandler
Requires Python 3 and TensorFlow 2

List of admitted input 1 types:
> 'vel'
> 'polarvel'

List of admitted input 2 types (both strings for position and velocity components need to be contained in input2_type):
> For the position component:
>> 'posabs'
>> 'posrel'
>> 'posinvdist'
>> 'polarposabs'
>> 'polarposrel'
>> 'polarposinvdist'
> For the velocity component:
>> 'velabs'
>> 'velrel'
>> 'polarvelabs'
>> 'polarvelreldiff'
>> 'polarvelrelcos'
"""
#!/usr/bin/env python
import sys
import numpy as np
import pickle as pkl
from numpy.linalg import norm
from scipy.linalg import hankel
from scipy.spatial.distance import pdist
from math import atan2, cos, sin

class DataHandler():
    def __init__(self, args):
        self.args = args
        self.tbpl = self.args.truncated_backprop_length # unused
        self.prev_horizon = self.args.prev_horizon
        self.prediction_horizon = self.args.prediction_horizon
        
        self.input1_type = self.args.input1_type
        self.input2_type = self.args.input2_type

        self.shuffle = True
        self.batch_traj_idx = 0
        self.batch_size = self.args.batch_size
        self.input1_batch = np.zeros( [self.batch_size, self.prev_horizon, self.args.input_dim] )
        self.input2_batch = np.zeros( [self.batch_size, self.prev_horizon, self.args.pedestrian_vector_dim] )
        self.target_batch = np.zeros( [self.batch_size, self.prediction_horizon, self.args.output_pred_state_dim] )
        
        self.trajectory_set = []
        """ 
        trajectory_set is a list of trajectories
        > Each trajectory is a list of dictionary variables representing each data point
        >> Each datapoint contains the follwing keys:
        >>> 'control_command': Command of the robot
        >>> 'future_positions': 
        >>> 'goal_position': Goal position of the robot
        >>> 'pedestrian_goal_position': Goal position of the pedestrian
        >>> 'pedestrian_state': Dictionary which contains the current 'position' and 'velocity' of the pedestrian
        >>> 'predicted_cmd': Not used
        >>> 'robot_state': Current position and orientation of the robot
        >>> 'robot_velocity': Current speed of the robot
        """
        
        self.file = None

    def load_data(self):
        min_length_trajectory = self.prev_horizon + self.prediction_horizon
        self.file = open(self.args.data_path + self.args.dataset, 'rb')
        tmp_trajectory_set = pkl.load(self.file, encoding='latin1')
        trajectory_set = []
        
        for traj in tmp_trajectory_set:
            sub_traj = []
            # Sub-sample trajectories to make dt = 0.4
            for t_id in range(0, len(traj), 4):
            # for t_id in range(0, len(traj), 2):
                sub_traj.append(traj[t_id])
            # Append zero velocities to goal position
            if traj:
                goal_data = traj[-1].copy()
                goal_data["pedestrian_state"]["velocity"] = np.array([0, 0])
                for t_id in range(10):
                    sub_traj.append(goal_data)
                if len(sub_traj) > min_length_trajectory:
                    trajectory_set.append(sub_traj)
        self.trajectory_set = trajectory_set

        input1_arrays = []
        input2_arrays = []
        Y_arrays = []
        
        for trajectory_idx in range(len(self.trajectory_set)):
            feedArrays = self.getTrajectoryAsBatch(trajectory_idx)
            input1_arrays.append( feedArrays['input'][0] )
            input2_arrays.append( feedArrays['input'][1] )
            Y_arrays.append( feedArrays['target'] )
            
        self.input1 = np.concatenate(input1_arrays, axis = 0)
        self.input2 = np.concatenate(input2_arrays, axis = 0)
        self.target = np.concatenate(Y_arrays, axis = 0)
        self.n_data_samples = self.target.shape[0]
        
        if self.shuffle:
            idxs = np.random.permutation(self.n_data_samples)
            self.input1 = self.input1[idxs]
            self.input2 = self.input2[idxs]
            self.target = self.target[idxs]    

    def getBatch(self):
        if self.batch_traj_idx + self.batch_size <= self.n_data_samples: # Regular batches
            self.input1_batch = self.input1[self.batch_traj_idx:self.batch_traj_idx+self.batch_size,:,:]
            self.input2_batch = self.input2[self.batch_traj_idx:self.batch_traj_idx+self.batch_size,:,:]
            self.target_batch = self.target[self.batch_traj_idx:self.batch_traj_idx+self.batch_size,:,:]
            self.batch_traj_idx += self.batch_size
            new_epoch = False
            
        else: # Last batch, fill with random data points
            remaining_samples = self.n_data_samples - self.batch_traj_idx
            self.input1_batch[0:remaining_samples, :, :] = self.input1[self.batch_traj_idx:, :, :]
            self.input2_batch[0:remaining_samples, :, :] = self.input2[self.batch_traj_idx:, :, :]
            self.target_batch[0:remaining_samples, :, :] = self.target[self.batch_traj_idx:, :, :]
            
            # Take random data points outside of the remaining ones in order to completely fill up the batches
            idxs = np.random.permutation(self.batch_traj_idx)[0:self.batch_size-remaining_samples] 
            self.input1_batch[remaining_samples:, :, :] = self.input1[idxs, :, :]
            self.input2_batch[remaining_samples:, :, :] = self.input2[idxs, :, :]
            self.target_batch[remaining_samples:, :, :] = self.target[idxs, :, :]
            
            self.batch_traj_idx = 0
            new_epoch = True

        return {"input": [self.input1_batch, self.input2_batch],
                "target": self.target_batch,
                "new_epoch": new_epoch}

    def getTrajectoryAsBatch(self, trajectory_idx):
        trajectory = self.trajectory_set[trajectory_idx]
        input1_seq_list = []
        input2_seq_list = []

        for sample in trajectory:
            input1_data = self.getInput1Data(sample)
            input2_data = self.getInput2Data(sample)

            input1_seq_list.append(input1_data)
            input2_seq_list.append(input2_data)
            
        vel_seq =np.stack(input1_seq_list, axis = 1) # (features, timesteps)
        other_seq = np.stack(input2_seq_list, axis = 1) # (features, timesteps)
        
        input1_feedArray = expand_sequence(vel_seq[:, 0:-self.prediction_horizon], self.prev_horizon)
        input2_feedArray = expand_sequence(other_seq[:, 0:-self.prediction_horizon], self.prev_horizon)
        target_feedArray = expand_sequence(vel_seq[:, self.prev_horizon:], self.prediction_horizon)
        
        return {
            'input': [input1_feedArray, input2_feedArray],
            'target': target_feedArray,
            'trajectory': trajectory
        }
        
    def getPaddedTrajectoryAsBatch(self, trajectory_idx):
        trajectory = self.trajectory_set[trajectory_idx]
        input1_seq_list = []
        input2_seq_list = []

        for sample in trajectory:            
            input1_data = self.getInput1Data(sample)
            input2_data = self.getInput2Data(sample)

            input1_seq_list.append(input1_data)
            input2_seq_list.append( input2_data )
        vel_seq =np.stack(input1_seq_list, axis = 1) # (features, timesteps)
        other_seq = np.stack(input2_seq_list, axis = 1) # (features, timesteps)

        padded_vel_seq = np.concatenate( [np.zeros((2, self.prev_horizon-1)), vel_seq], axis = 1 )
        padded_other_seq = np.concatenate( [np.zeros((4, self.prev_horizon-1)), other_seq], axis = 1 )

        input1_feedArray = expand_sequence(padded_vel_seq, self.prev_horizon)
        input2_feedArray = expand_sequence(padded_other_seq, self.prev_horizon)

        return {
            'input': [input1_feedArray, input2_feedArray],
            'trajectory': trajectory
        }
        
    def getInput1Data(self, sample):
        ped_vel = sample['pedestrian_state']['velocity']
        
        if self.input1_type == "vel": # Velocity in cartesian coordinates
            # Velocity as data
            data = ped_vel
        elif self.input1_type == "polarvel": # Velocity in polar coordinates
            # Velocity modulus and direction as data
            modulus = norm(ped_vel)
            direction = atan2(ped_vel[1], ped_vel[0])
            data = np.array([modulus, direction])
        else:
            raise Exception("No valid type for Input 1")
        
        return data
    
    def getInput2Data(self, sample):
        ped_pos = sample['pedestrian_state']['position']
        ped_vel = sample['pedestrian_state']['velocity']
        robot_pos = sample['robot_state'][0:2]
        robot_vel = sample['robot_velocity']
        
        # Position component
        if "polarpos" in self.input2_type: # Position in polar coordinates
            if "posabs" in self.input2_type:
                modulus = norm(robot_pos)
                direction = atan2(robot_pos[1], robot_pos[0])
                pos = np.array([modulus, direction])
            elif "posrel" in self.input2_type:
                distance = robot_pos - ped_pos
                modulus = norm(distance)
                direction = atan2(distance[1], distance[0])
                pos = np.array([modulus, direction])
            elif "posinvdist" in self.input2_type:
                distance = robot_pos - ped_pos
                invdist = np.zeros(2)
                invdist[0] = 1/(distance[0] + 1)
                invdist[1] = 1/(distance[1] + 1)
                modulus = norm(invdist)
                direction = atan2(invdist[1], invdist[0])
                pos = np.array([modulus, direction])
            else:
                raise Exception("No valid position-component type for Input 2")
            
        else: # Position in cartesian coordinates
            if "posabs" in self.input2_type:
                pos = robot_pos
            elif "posrel" in self.input2_type:
                pos = robot_pos - ped_pos
            elif "posinvdist" in self.input2_type:
                distance = robot_pos - ped_pos
                invdist = np.zeros(2)
                invdist[0] = 1/(distance[0] + 1)
                invdist[1] = 1/(distance[1] + 1)
                pos = invdist
            else:
                raise Exception("No valid position-component type for Input 2")
        
        # Velocity component
        if "polarvel" in self.input2_type: # Velocity in polar coordinates
            if "velabs" in self.input2_type: # Polar absolute velocity
                modulus = norm(robot_vel)
                direction = atan2(robot_vel[1], robot_vel[0])
                vel = np.array([modulus, direction])
            elif "velreldiff" in self.input2_type: # Polar relative velocity  
                modulus = norm(robot_vel - ped_vel)
                direction_rob = atan2(robot_vel[1], robot_vel[0])
                direction_ped = atan2(robot_vel[1], robot_vel[0])
                angular_distance = direction_rob - direction_ped
                if angular_distance > 180:
                    angular_distance -= 360
                elif angular_distance < -180:
                    angular_distance += 360 
                vel = np.array([modulus, angular_distance])
            elif "velrelcos" in self.input2_type: # Polar relative velocity using cosine distance
                modulus = norm(robot_vel - ped_vel)
                distance = pdist([robot_vel, ped_vel], 'cosine')[0]
                if np.isnan(distance):
                    distance = 0
                vel = np.array([modulus, distance])
            else:
                raise Exception("No valid velocity-component type for Input 2")
                
        else: # Velocity in cartesian coordinates
            if "velabs" in self.input2_type:
                vel = robot_vel
            elif "velrel" in self.input2_type:
                vel = robot_vel - ped_vel
            else:
                raise Exception("No valid velocity-component type for Input 2")
        
        data =  np.concatenate( [pos, vel] )
        
        return data


def prediction_to_trajectory(initial_position, prediction, args):
    dt = args.dt
    output_type = args.input1_type
    
    trajectory_list = [ initial_position ]

    for pred_step in prediction:
        if output_type == "vel": # Velocity in cartesian coordinates
            new_point = trajectory_list[-1] + pred_step * dt
            
        elif output_type == "polarvel": # Velocity in polar coordinates
            modulus = pred_step[0]
            direction = pred_step[1]
            new_point = trajectory_list[-1] + modulus * np.array((cos(direction), sin(direction))) * dt
        else:
            raise Exception("Unimplemented output type")
        
        trajectory_list.append(new_point)
    trajectory = np.stack(trajectory_list, axis = 0)
    return trajectory

def expand_sequence(sequence_array, horizon): 
    # Sequence array has shape [features, time steps]
    # Expanded sequence will have shape [time steps, horizon, features]
    expanded_sequence = np.zeros((sequence_array.shape[1]-horizon+1,\
                                  horizon,\
                                  sequence_array.shape[0]))
    
    for i in range(sequence_array.shape[0]): # For each feature
        sequence = sequence_array[i, :]
        expanded_sequence[:, :, i] = hankel(sequence[0:horizon],\
                                            sequence[horizon-1:]).transpose()
    
    return expanded_sequence

def reduce_sequence(hankel_matrix):
    aux = []
    for feature_idx in range(hankel_matrix.shape[2]):
        aux.append( np.concatenate([hankel_matrix[0, :, feature_idx], hankel_matrix[1:, -1, feature_idx]], axis = 0) )
    return np.stack(aux, axis = 1)
