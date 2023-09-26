#!/usr/bin/env python
import sys
import numpy as np
# from scipy.linalg import hankel

if sys.version_info[0] < 3:
    sys.path.append('../src/data_utils')
    import Support as sup
    from DataHandler import *
else:
    import src.data_utils.Support as sup
    from src.data_utils.DataHandler import *
    
class DataHandlerLSTM_Keras(DataHandler):
    def __init__(self, args):
        super().__init__(args)
        # self.tbpl = self.prev_horizon+1
    
    def getTestTrajectoryAsBatch(self, trajectory_idx, max_sequence_length=1000):
        """
        Get a trajectory out of the trajectory set in the same format as for the standard training data
        (e.g. for validation purposes).
        """
        print(trajectory_idx)
        id = self.test_trajectory_set[trajectory_idx][0]
        traj = self.test_trajectory_set[trajectory_idx][1]

        sequence_length = min(max_sequence_length, traj.pose_vec.shape[0] - self.output_sequence_length-self.prev_horizon)
        batch_x = np.zeros([1, sequence_length, (self.prev_horizon+1)*self.input_dim])  # data fed for training
        batch_vel = np.zeros([1, sequence_length, (self.prev_horizon + 1) * self.input_state_dim])
        batch_grid = np.zeros([1, sequence_length, int(np.ceil(self.submap_width / self.args.submap_resolution)), int(np.ceil(self.submap_height / self.args.submap_resolution))])
        batch_goal = np.zeros([1, sequence_length, 2])
        batch_y = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
        batch_pos = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
        pedestrian_grid = np.zeros([1, sequence_length, self.pedestrian_vector_dim])
        other_agents_pos = self.fillBatch(id, 0, self.prev_horizon, sequence_length, batch_x, batch_vel,batch_pos,batch_grid, pedestrian_grid, batch_goal, batch_y, traj, centered_grid=self.centered_grid,testing=True)

        vel_seq = 
        other_agents_info_seq = 

        # return batch_x, batch_vel, batch_pos,batch_goal, batch_grid, pedestrian_grid, batch_y, other_agents_pos, traj

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
		
    
    def fillBatch(self, agent_id, batch_idx, start_idx, truncated_backprop_length, batch_x, batch_vel,batch_pos,batch_grid, pedestrian_grid, batch_goal, batch_y, trajectory, centered_grid=False,testing=False):
		"""
		Fill the data batches of batch_idx with data for all truncated backpropagation steps.
		"""
		other_agents_pos = []

		for tbp_step in range(truncated_backprop_length):

			# Input values
			query_time = trajectory.time_vec[start_idx + tbp_step]
			for prev_step in range(self.prev_horizon,-1,-1):
				current_pos = np.array([trajectory.pose_vec[start_idx + tbp_step - prev_step, 0], trajectory.pose_vec[
					                        start_idx + tbp_step - prev_step, 1]])
				current_vel = np.array([trajectory.vel_vec[start_idx + tbp_step - prev_step, 0] * self.norm_const_vx, trajectory.vel_vec[
					                        start_idx + tbp_step - prev_step, 1] * self.norm_const_vy])

				if self.args.normalize_data:
					self.normalize_pos(current_pos)
					self.normalize_vel(current_vel)

				batch_x[batch_idx, tbp_step, prev_step*self.input_dim:(prev_step+1)*self.input_dim] = np.array([current_pos[0],
																										current_pos[1],
																										current_vel[0],
																										current_vel[1]])
				batch_vel[batch_idx, tbp_step, prev_step*self.input_state_dim:(prev_step+1)*self.input_state_dim] = np.array([current_vel[0],
																											current_vel[1]])

			heading = math.atan2(current_vel[1], current_vel[0])
			if not testing:
				if centered_grid:
					grid_center = current_pos
					grid = self.agent_container.occupancy_grid.getSubmapByCoords(grid_center[0],
																																			 grid_center[1],
																																			 self.submap_width, self.submap_height)
				else:
					grid = self.batch_sequences[batch_idx][1]
					grid_center = trajectory.pose_vec[0, 0:2]

				if self.rotated_grid:
					grid = sup.rotate_grid_around_center(grid, heading * 180 / math.pi)  # rotation in degrees
				else:
					heading = 0
					grid = sup.rotate_grid_around_center(grid, heading * 180 / math.pi)  # rotation in degrees

				batch_grid[batch_idx, tbp_step, :, :] = grid

			batch_goal[batch_idx, tbp_step, :] = trajectory.goal
			# Find positions of other pedestrians at the current timestep

			other_poses = trajectory.other_agents_positions[start_idx + tbp_step]
			n_other_agents = other_poses.shape[0]
			other_velocities = trajectory.other_agents_velocities[start_idx + tbp_step]
			other_agents_pos.append(other_poses)
			other_pos_local_frame = sup.positions_in_local_frame(current_pos, heading, other_poses)
			try:
				if self.args.relative_info:
					radial_pedestrian_grid = np.zeros([self.pedestrian_vector_dim])
					for ag_id in range(n_other_agents):
						radial_pedestrian_grid[ag_id*4:ag_id*4+4] = np.array([other_poses[ag_id,0] - current_pos[0],
						                                                      other_poses[ag_id, 1] - current_pos[1],
						                                                      other_velocities[ag_id, 0] - current_vel[0],
						                                                      other_velocities[ag_id, 1] - current_vel[1]])
					for ag_id in range(n_other_agents,int(self.args.pedestrian_vector_dim/4)):
						radial_pedestrian_grid[ag_id*4:ag_id*4+4] = np.array([100,100,0,0])

				else:
					radial_pedestrian_grid = sup.compute_radial_distance_vector(self.pedestrian_vector_dim, other_pos_local_frame,
																																max_range=self.max_range_ped_grid, min_angle=0, max_angle=2*np.pi,
																																normalize=True)
			except:
				radial_pedestrian_grid = sup.compute_radial_distance_vector(self.pedestrian_vector_dim, other_pos_local_frame,
				                                                            max_range=self.max_range_ped_grid, min_angle=0,
				                                                            max_angle=2 * np.pi,
				                                                            normalize=True)

			pedestrian_grid[batch_idx, tbp_step, :] = radial_pedestrian_grid

			try:
				rel_pos = np.array([other_poses[0,0] - current_pos[0],other_poses[0, 1] - current_pos[1]])
				rel_vel = np.array([other_velocities[0, 0] - current_vel[0],
							              other_velocities[0, 1] - current_vel[1]])
			except:
				rel_pos = np.array([-100,-100])
				rel_vel = np.array([0,0])

			if (self.args.pedestrian_vector_dim == 4):
				pedestrian_grid[batch_idx, tbp_step, :] = np.concatenate((rel_pos, rel_vel), axis=0)
			elif (self.args.pedestrian_vector_dim == 3):
				pedestrian_grid[batch_idx, tbp_step, :] = np.concatenate((np.linalg.norm(rel_pos), rel_vel), axis=0)
			else:
				pedestrian_grid[batch_idx, tbp_step, :] = radial_pedestrian_grid

			# Output values
			for pred_step in range(self.output_sequence_length):
				vx = trajectory.vel_vec[start_idx + tbp_step + 1 + pred_step, 0]
				vy = trajectory.vel_vec[start_idx + tbp_step + 1 + pred_step, 1]
				px = trajectory.pose_vec[start_idx + tbp_step + 1 + pred_step, 0]
				py = trajectory.pose_vec[start_idx + tbp_step + 1 + pred_step, 1]
				batch_y[batch_idx, tbp_step, self.output_state_dim*pred_step] = vx
				batch_y[batch_idx, tbp_step, self.output_state_dim*pred_step + 1] = vy
				if self.args.normalize_data:
					self.normalize_vel(batch_y[batch_idx, tbp_step, self.output_state_dim*pred_step:self.output_state_dim*pred_step+2])
				batch_pos[batch_idx, tbp_step, self.output_state_dim*pred_step] = px
				batch_pos[batch_idx, tbp_step, self.output_state_dim*pred_step + 1] = py


		return other_agents_pos