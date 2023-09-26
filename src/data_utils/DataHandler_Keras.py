#!/usr/bin/env python
import sys
import numpy as np
from scipy.linalg import hankel

if sys.version_info[0] < 3:
    sys.path.append('../src/data_utils')
    import Support as sup
    from DataHandler import *
else:
    import src.data_utils.Support as sup
    from src.data_utils.DataHandlerLSTM import *

class DataHandler_Keras(DataHandlerLSTM):
    def __init__(self, args):
        super().__init__(args)
        self.tbpl = self.prev_horizon+1
        
        # Batches size redefinition
        self.batch_grid = np.zeros((self.batch_size, self.prev_horizon+1, int(self.args.submap_width / self.args.submap_resolution),
                                    int(self.args.submap_height / self.args.submap_resolution)))
        self.batch_x = np.zeros((self.batch_size, self.prev_horizon+1, self.args.input_dim))
        self.batch_vel = np.zeros((self.batch_size, self.prev_horizon+1, self.args.input_state_dim))
        self.batch_pos = np.zeros((self.batch_size, self.prev_horizon+1, self.args.input_state_dim))
        self.batch_goal = np.zeros((self.batch_size, self.prev_horizon+1, 2))
        self.other_agents_info = np.zeros((self.batch_size, self.prev_horizon+1, self.args.pedestrian_vector_dim))
        self.other_agents_pos = np.zeros((self.batch_size, self.prev_horizon+1, self.args.pedestrian_vector_state_dim))
        self.other_agents_vel = np.zeros((self.batch_size, self.prev_horizon+1, self.args.pedestrian_vector_state_dim))
        self.batch_y = np.zeros((self.batch_size, self.prediction_horizon, self.args.output_dim))
        self.pedestrian_grid = np.zeros([self.batch_size, self.tbpl, self.pedestrian_vector_dim])

    def getBatch(self):
        """
        		Get the next batch of training data.
        		"""
        # Update sequences
        # If batch sequences are empty and need to be filled
        trajectory = []
        if len(self.batch_sequences) == 0:
            for b in range(0, min(self.batch_size, len(self.trajectory_set))):
                id, trajectory = self.trajectory_set[self.data_idx]
                self.data_idx += 1
                self.batch_sequences.append(trajectory)
                self.batch_ids.append(id)
        # If batch sequences are filled and can be used or need to be updated.
        other_agents_pos = []
        new_epoch = False
        for ii, traj in enumerate(self.batch_sequences):
            if self.sequence_idx[ii] + self.tbpl + self.output_sequence_length + 1 >= len(traj):
                id, trajectory = self.trajectory_set[self.data_idx]
                self.data_idx = (self.data_idx + 1) % int(len(self.trajectory_set) * self.train_set)
                if self.data_idx == 0:
                    new_epoch = True
                self.batch_sequences[ii] = trajectory
                self.batch_ids[ii] = id
                self.sequence_idx[ii] = self.args.prev_horizon
                self.sequence_reset[ii] = 1
            else:
                self.sequence_reset[ii] = 0

        # Fill the batch
        other_agents_pos = []
        for ii in range(0, min(self.batch_size, len(self.trajectory_set) - len(self.trajectory_set) % self.batch_size)):
            traj = self.batch_sequences[ii]
            agent_id = self.batch_ids[ii]
            other_agents_pos.append(
                self.fillBatch(agent_id, ii, int(self.sequence_idx[ii]), self.tbpl, self.batch_x, self.batch_vel,
                               self.batch_pos, self.batch_grid, self.pedestrian_grid, self.batch_goal, self.batch_y,
                               traj, centered_grid=self.centered_grid))
            self.sequence_idx[ii] += self.tbpl
        
        batch = {
            'x': self.batch_x,
            'vel': self.batch_vel,
            'pos': self.batch_pos,
            'goal': self.batch_goal,
            'grid': self.batch_grid,
            'other_agents_info': self.other_agents_info,
            'y': self.batch_y,
            'other_agents_pos': self.other_agents_pos,
            'other_agents_vel': self.other_agents_vel,
            'new_epoch': new_epoch
        }
        return batch

    def fillBatch(self, agent_id, batch_idx, start_idx, truncated_backprop_length, batch_x, batch_vel, batch_pos,
                  batch_grid, pedestrian_grid, batch_goal, batch_y, trajectory, centered_grid=False):
        """
				Fill the data batches of batch_idx with data for all truncated backpropagation steps.
				"""

        for prev_step in range(self.prev_horizon,-1,-1):

            # Input values
            current_pos = np.array([trajectory.pose_vec[start_idx - prev_step, 0], trajectory.pose_vec[start_idx - prev_step, 1]])
            current_vel = np.array([trajectory.vel_vec[start_idx - prev_step, 0] , trajectory.vel_vec[start_idx - prev_step, 1]])

            if self.args.normalize_data:
                self.normalize_pos(current_pos)
                self.normalize_vel(current_vel)

            batch_x[batch_idx, prev_step, :] = np.array([current_pos[0],current_pos[1],current_vel[0],current_vel[1]])
            batch_vel[batch_idx, prev_step, :] = np.array([current_vel[0],current_vel[1]])

            heading = math.atan2(current_vel[1], current_vel[0])

            # Find positions of other pedestrians at the current timestep
            other_poses = trajectory.other_agents_positions[start_idx - prev_step]
            other_agents_pos = other_poses
            n_other_agents = other_poses.shape[0]
            if n_other_agents>0:
                other_velocities = trajectory.other_agents_velocities[start_idx - prev_step]
                other_pos_local_frame = sup.positions_in_local_frame(current_pos, heading, other_poses)

                # TODO: it only works for one agent now
                rel_pos = other_poses - current_pos
                rel_vel = other_velocities - current_vel
                distance = np.linalg.norm(rel_pos)
                pedestrian_grid[batch_idx, prev_step, :] = np.concatenate(
                  (np.array([np.linalg.norm(rel_pos)]), rel_vel[0]))

        # Output values
        for pred_step in range(self.output_sequence_length):
            vx = trajectory.vel_vec[start_idx +1+ pred_step, 0]
            vy = trajectory.vel_vec[start_idx +1+ pred_step, 1]
            px = trajectory.pose_vec[start_idx +1+ pred_step, 0]
            py = trajectory.pose_vec[start_idx +1+ pred_step, 1]
            batch_y[batch_idx, pred_step, 0] = vx
            batch_y[batch_idx, pred_step, 1] = vy
            if self.args.normalize_data:
                self.normalize_vel(batch_y[batch_idx, pred_step,:])
            batch_pos[batch_idx, pred_step, 0] = px
            batch_pos[batch_idx, pred_step, 1] = py

        return other_agents_pos
    
    def getTrajectoryAsBatch(self, trajectory_idx, max_sequence_length=1000,unit_testing=False):
        """
        Get a trajectory out of the trajectory set in the same format as for the standard training data
        (e.g. for validation purposes).
        """
        if unit_testing:
            traj = self.test_trajectory_set[trajectory_idx][1]
        else:
            traj = self.trajectory_set[trajectory_idx][1]

        sequence_length = min(max_sequence_length, len(traj) - self.prediction_horizon) - self.prev_horizon
        
        # Old data structures, used for plotting
        batch_x = np.zeros([sequence_length,self.prev_horizon+1, self.args.input_dim])  # data fed for training
        batch_pos = np.zeros([sequence_length,self.prev_horizon+1, self.args.input_state_dim])  # data fed for training
        batch_vel = np.zeros([sequence_length,self.prev_horizon+1, self.args.input_state_dim])  # data fed for training
        batch_goal = np.zeros([sequence_length, 2])
        batch_target = np.zeros([sequence_length, self.prediction_horizon,self.args.output_dim ])

        other_agents_info = np.zeros([sequence_length, self.prev_horizon+1, self.args.pedestrian_vector_dim])
        batch_grid = np.zeros((sequence_length, int(self.args.submap_width / self.args.submap_resolution),
                            int(self.args.submap_height / self.args.submap_resolution)))
        n_other_agents =  traj.other_agents_positions[0].shape[0]
        other_agents_pos = np.zeros((sequence_length, n_other_agents,self.args.pedestrian_vector_state_dim))
        other_agents_vel = np.zeros((sequence_length, self.args.pedestrian_vector_state_dim))
        
        # New data structures, fed to the network
        batch_vel_seq = np.zeros([sequence_length, self.prev_horizon+1, self.args.input_state_dim])
        other_agents_info_seq = np.zeros([sequence_length, self.prev_horizon+1, self.args.pedestrian_vector_dim])

        for batch_idx in range(sequence_length):
            other_agents_pos[batch_idx,:,:] = self.fillBatch(id, batch_idx, self.prev_horizon+batch_idx, sequence_length, batch_x, batch_vel, batch_pos,
                                              batch_grid, other_agents_info, batch_goal, batch_target, traj,
                                              centered_grid=self.centered_grid)

        batch = {
            'x': batch_x,
            'vel': batch_vel,
            'pos': batch_pos,
            'goal': batch_goal,
            'grid': batch_grid,
            'other_agents_info': other_agents_info,
            'y': batch_target,
            'other_agents_pos': other_agents_pos,
            'other_agents_vel': other_agents_vel,
            'traj': traj,
            'vel_seq': batch_vel_seq,
            'other_agents_info_seq': other_agents_info_seq
        }
        return batch
        
        
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
