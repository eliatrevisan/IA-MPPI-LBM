#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as pl
import pickle as pkl
import tensorflow as tf

if sys.version_info[0] < 3:
	import yaml

	sys.path.append('../src/data_utils')
	sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
	import Support as sup
	import OccupancyGrid as occ
	import MemoryBuffer
	import rospy
	from geometry_msgs.msg import Pose
	from geometry_msgs.msg import Twist
	from pedsim_msgs.msg import TrackedPersons
	from sensor_msgs.msg import LaserScan
	from nav_msgs.msg import OccupancyGrid
	from nav_msgs.msg import Path
	import rosbag
else:
	sys.path.append('../src/data_utils')
	import src.data_utils.Support as sup
	import src.data_utils.MemoryBuffer as MemoryBuffer
	import src.data_utils.OccupancyGrid as occ
import random
import os

from multiprocessing.pool import ThreadPool
import math
from time import sleep
from parser import *
import argparse

tfk = tf.keras


class DataHandler():

	def __init__(self, args):
		self.args = args
		self.control_command = np.empty([1, 2])
		self.laser_data = np.empty([1, 720])
		self.grid = np.empty([120, 120])
		self.pedestrian = []
		self.trajectory = []
		self.pedestrian_state = []
		self.robot_command_set = []
		self.robot_state = []
		self.robot_pred_cmd_set = []
		self.data = {}
		self.trajectory_set = []
		self.data_idx = 0
		self.dt = 0.1
		self.batch_size = self.args.batch_size
		self.tbpl = self.args.truncated_backprop_length
		self.cmd_vector_dim = self.args.cmd_vector_dim
		self.prev_horizon = self.args.prev_horizon
		self.prediction_horizon = self.args.prediction_horizon
		self.pred_cmd = np.empty([1, self.prediction_horizon * self.cmd_vector_dim])
		self.min_length_trajectory = self.tbpl + 1 + self.prediction_horizon * 2 + args.prev_horizon
		# Batches
		self.batch_cmd = np.zeros((self.batch_size, self.tbpl, self.cmd_vector_dim))
		self.batch_pred_cmd = np.zeros((self.batch_size, self.tbpl, self.prediction_horizon * self.cmd_vector_dim))
		self.batch_robot_state = np.zeros((self.batch_size, self.tbpl, 3))
		self.batch_grid = np.zeros((self.batch_size, self.tbpl, int(self.args.submap_width / self.args.submap_resolution),
		                            int(self.args.submap_height / self.args.submap_resolution)))
		self.batch_x = np.zeros((self.batch_size, self.tbpl, self.args.input_dim*(self.prev_horizon+1)))
		self.batch_vel = np.zeros((self.batch_size, self.tbpl, self.args.input_state_dim*(self.prev_horizon+1)))
		self.batch_pos = np.zeros((self.batch_size, self.tbpl, self.args.input_state_dim*(self.prev_horizon+1)))
		self.batch_goal = np.zeros((self.batch_size, self.tbpl, 2))
		self.other_agents_info = np.zeros((self.batch_size, self.tbpl, self.args.pedestrian_vector_dim))
		self.other_agents_pos = np.zeros((self.batch_size, self.tbpl, self.args.pedestrian_vector_state_dim))
		self.batch_enc = np.zeros((self.batch_size, self.prev_horizon, self.args.input_dim))
		self.batch_dec = np.zeros((self.batch_size, self.prediction_horizon, self.args.output_dim))
		self.batch_target = np.zeros((self.batch_size, self.tbpl, self.args.output_dim * self.prediction_horizon))
		self.batch_sequences = []
		# indicates whether sequences are reset (hidden state of LSTMs needs to be reset accordingly)
		self.sequence_reset = np.ones([self.batch_size])
		self.sequence_idx = np.zeros([self.batch_size])+self.args.prev_horizon

		# Replay buffer for oline training
		self.buffer = MemoryBuffer.MemoryBuffer(self.args.min_buffer_size, self.args.max_buffer_size)

		self.max_traj = self.args.max_trajectories

		self.global_fig = pl.figure("occupancy grid ")
		self.ax_grid = pl.subplot()
		"""
		print("Extracting the occupancy grid ...")
		# Occupancy grid data
		self.occupancy_grid = occ.OccupancyGrid()
		temp = self.args.data_path + '/map.npy'
		# Only used to create a map from png
		map_args = {"file_name": self.args.file_name,
								"resolution": self.args.resolution,
								"map_size": np.array([self.args.map_size_x,self.args.map_size_y]), }
		if not os.path.isfile(self.args.data_path + '/map.npy'):
				print("Creating map from png ...")
				sup.create_map_from_png(data_path=self.args.data_path, **map_args)

		map_data = np.load(os.path.join(self.args.data_path, 'map.npy'), encoding='latin1', allow_pickle=True)
		self.occupancy_grid.gridmap = map_data.item(0)['Map']  # occupancy values of cells
		self.occupancy_grid.resolution = map_data.item(0)['Resolution']  # map resolution in [m / cell]
		self.occupancy_grid.map_size = map_data.item(0)['Size']  # map size in [m]
		self.occupancy_grid.center = self.occupancy_grid.map_size / 2.0

		# FIlter all -1 values present on MAP to zero.
		for ii in range(self.occupancy_grid.gridmap.shape[0]):
				for jj in range(self.occupancy_grid.gridmap.shape[1]):
						if self.occupancy_grid.gridmap[ii, jj] == -1:
								self.occupancy_grid.gridmap[ii, jj] = 0.0
		"""
		if sys.version_info[0] < 3:
			config = yaml.load(open(self.args.topics_config))  # read the yaml file and get the required parameters

			self.robot_topic = config['robot_state_topic']
			self.pedestrian_topic = config['pedestrian_topic']
			self.commmand_topic = config['command_topic']
			self.laser_topic = config['laser_topic']
			self.grid_topic = config['grid_topic']
			self.predict_topic = config['predict_topic']
			self.rate = config['rate']
			rospy.Subscriber(self.robot_topic, Pose, self.RobotStateCB)  # subscribe to various topics
			rospy.Subscriber(self.commmand_topic, Twist, self.ControlCommandCB)
			rospy.Subscriber(self.laser_topic, LaserScan, self.LaserDataCB)
			rospy.Subscriber(self.pedestrian_topic, TrackedPersons, self.TrackedPersonsCB)
			rospy.Subscriber(self.grid_topic, OccupancyGrid, self.GridProcessingCB)
			rospy.Subscriber(self.predict_topic, Path, self.PredictedCommandCB)

	def process_simulation_data_(self):
		# load data from csv for offline training

		# read the required csv files
		if os.path.exists(self.args.data_path):

			if os.path.exists(os.path.join(self.args.data_path, self.args.bag_file)):
				bag = rosbag.Bag(os.path.join(self.args.data_path, self.args.bag_file))

				# variables for storing robot control commands
				linear_vel = np.empty([1])
				angular_vel = np.empty([1])

				# get the control command data
				data_idx = 0
				for topic, msg, t in bag.read_messages(topics=['jackal_velocity_controller/cmd_vel']):
					linear_vel[:] = msg.linear.x
					angular_vel[:] = msg.angular.z
					control_command = np.concatenate([linear_vel, angular_vel])  # [1,2]
					self.robot_command_set.append(control_command.copy())
					data_idx += 1

				# convert from list to numpy array
				self.robot_command_set = np.array(self.robot_command_set)

				for topic, msg, t in bag.read_messages(topics=['/predicted_cmd']):
					self.args.prediction_cmd_horizon = len(msg.poses)
					cmd_pred = np.zeros(self.args.prediction_cmd_horizon * 2)
					for n in range(self.args.prediction_cmd_horizon):
						linear_vel = msg.poses[n].pose.position.x
						angular_vel = msg.poses[n].pose.position.y

						if math.isnan(angular_vel):
							angular_vel = 0

						if math.isnan(linear_vel):
							linear_vel = 0
						cmd_pred[2 * n] = linear_vel
						cmd_pred[2 * n + 1] = angular_vel
					self.robot_pred_cmd_set.append(cmd_pred.copy())

				# convert from list to numpy array
				self.robot_pred_cmd_set = np.asanyarray(self.robot_pred_cmd_set)

				# variables storing robot pose data
				position = np.empty([2])
				orientation = np.empty([1])
				self.robot_state = []

				for topic, msg, t in bag.read_messages(topics=['/robot_state']):
					position[:] = [msg.position.x, msg.position.y]
					orientation[:] = msg.orientation.z
					robot_pose = np.concatenate([position, orientation])  # [1,3]
					self.robot_state.append(robot_pose)

				# convert from list to numpy array
				self.robot_state = np.asanyarray(self.robot_state)

				# check the number of times the robot reaches the goal state, hence providing the number of trajectories.
				self.robot_trajectory_number = 0
				for i in range(1, self.robot_state.shape[0]):
					dist = np.linalg.norm(self.robot_state[i, :2] - self.robot_state[i - 1, :2])
					if dist > self.args.distance_threshold:  # trajectory completion condition
						self.robot_trajectory_number += 1

				print("total trajectories: {}".format(self.robot_trajectory_number))

				for topic, msg, t in bag.read_messages(topics=['/pedsim_visualizer/tracked_persons']):
					# number of pedestrians in recorded data
					ped_number = len(msg.tracks)
					pedestrian = []

					for ped_idx in range(ped_number):
						ped_data = {
							"position": [msg.tracks[ped_idx].pose.pose.position.x, msg.tracks[ped_idx].pose.pose.position.y],
							"velocity": [msg.tracks[ped_idx].twist.twist.linear.x, msg.tracks[ped_idx].twist.twist.linear.y]
							}
						pedestrian.append(ped_data)
					self.pedestrian_state.append(pedestrian)

				# convert from list to numpy array
				# self.pedestrian_state = np.asanyarray(self.pedestrian_state)

				# obtain the training data with the necessary data assigned for each trajectory
				traj = []
				for j in range(np.min(
						[self.robot_state.shape[0], self.robot_pred_cmd_set.shape[0], len(self.pedestrian_state),
						 self.robot_command_set.shape[0]])):  # iterate over trajectory steps

					data = {"robot_state": self.robot_state[j, :],
					        "pedestrian_state": self.pedestrian_state[j],
					        "control_command": self.robot_command_set[j, :],
					        "pred_command": self.robot_pred_cmd_set[j, :]
					        }

					traj.append(data)

					# break the loop when ever a goal state is encountered, signalling the end of data storage
					# for current trajectory
					if j > 1:
						dist = np.linalg.norm(self.robot_state[j, :2] - self.robot_state[j - 1, :2])
						if dist > self.args.distance_threshold:  # trajectory completion condition
							self.trajectory_set.append(traj)
							traj = []

				bag.close()
				random.shuffle(self.trajectory_set)  # shuffle the final dataset
			else:
				print("Could not find bag file...")

	def RobotStateCB(self, msg):

		pos_x = msg.position.x  # get the current state of the robot
		pos_y = msg.position.y
		orientation = msg.orientation.z
		self.robot_state = np.array([[pos_x, pos_y, orientation]])

	def ControlCommandCB(self, msg):

		linear_vel = msg.linear.x  # get the output commands from LMPCC
		angular_vel = msg.angular.z
		self.control_command = np.array([[linear_vel, angular_vel]])

	def LaserDataCB(self, msg):

		laser_data = []  # store the LiDAR data
		laser_data[:] = msg.ranges

		for i in range(len(laser_data)):
			if np.isinf(laser_data[i]):
				laser_data[i] = 30.0
		self.laser_data = np.array([laser_data])

	def TrackedPersonsCB(self, msg):
		self.pedestrian_state = []
		pos = np.zeros((1, 2))
		vel = np.zeros((1, 2))
		for person in range(len(msg.tracks)):  # for each detected pedestrian, get the required data
			pose_x = msg.tracks[person].pose.pose.position.x
			pose_y = msg.tracks[person].pose.pose.position.y
			vel_x = msg.tracks[person].twist.twist.linear.x
			vel_y = msg.tracks[person].twist.twist.linear.y
			pos = np.vstack((pos, [pose_x, pose_y]))
			vel = np.vstack((vel, [vel_x, vel_y]))
			ped_data = {"position": pos[1:],
			            "velocity": vel[1:]
			            }
		self.pedestrian = ped_data

	def GridProcessingCB(self, msg):
		data = []
		data[:] = msg.data
		for i in range(len(data)):  # binarize the static grid data
			if data[i] != 0:
				data[i] = 1
			else:
				data[i] = 0
		data = np.array([data])
		data = np.reshape(data, (msg.info.width, msg.info.height))
		self.grid = self.grid_rotate(data)

	def PredictedCommandCB(self, msg):

		pred_cmd = np.zeros(self.args.prediction_cmd_horizon * 2)
		horizon = len(msg.poses)

		for n in range(horizon):
			linear_vel = msg.poses[n].pose.position.x
			angular_vel = msg.poses[n].pose.position.y

			if math.isnan(angular_vel):
				angular_vel = 0
			if math.isnan(linear_vel):
				linear_vel = 0
			pred_cmd[2 * n] = linear_vel
			pred_cmd[2 * n + 1] = angular_vel

		self.pred_cmd = np.array([pred_cmd])

	def collect_data(self):

		if self.control_command.size != 0 and self.pred_cmd.size != 0:
			self.data = {"robot_state": self.robot_state,
			             "pedestrian_state": self.pedestrian,
			             "control_command": self.control_command,
			             "predicted_cmd": self.pred_cmd
			             }

			self.trajectory.append(self.data)

			if len(self.trajectory) > 1:
				dist = np.linalg.norm(self.trajectory[-1]["robot_state"][:, :2] - self.trajectory[-2]["robot_state"][:, :2])
				if dist > self.args.distance_threshold:
					self.trajectory_set.append(self.trajectory)
					self.trajectory = []

		print("Dataset length: " + str(len(self.trajectory_set)))
		print("Trajectory length" + str(len(self.trajectory)))

	def getSeqToSeqBatch(self):
		"""
		Get the next batch of training data.
		"""
		# Update sequences
		# If batch sequences are filled and can be used or need to be updated.
		new_epoch = False
		t_idx = self.prev_horizon
		for ii in range(0, self.batch_size):
			traj = self.trajectory_set[self.data_idx]
			if self.sequence_idx[ii] + self.prediction_horizon + 1 >= len(traj):  # change trajectory
				self.data_idx = (self.data_idx + 1) % int(len(self.trajectory_set))
				if self.data_idx == 0:
					new_epoch = True
				self.sequence_idx[ii] = self.prev_horizon
				self.sequence_reset[ii] = 1
				t_idx = self.prev_horizon
			else:
				# start_index = 0 + self.prev_horizon
				# end_index = len(traj) - self.prediction_horizon
				# for t_idx in range(start_index,end_index):
				idx = 0
				for i in range(t_idx - self.prev_horizon, t_idx + 1, 1):
					robot_pos = traj[i]["robot_state"][:2]
					robot_vel = traj[i]["control_command"][:]
					ped_pos = traj[i]["pedestrian_state"]["position"]
					ped_vel = traj[i]["pedestrian_state"]["velocity"]
					self.batch_enc[ii, idx, :] = np.concatenate((robot_pos, robot_vel, ped_pos, ped_vel))
					idx += 1
				idx = 0
				for i in range(t_idx, t_idx + self.prediction_horizon, 1):
					self.batch_dec[ii, idx, :] = traj[i - 1]["pedestrian_state"]["velocity"]
					self.batch_dec_target[ii, idx, :] = traj[i]["pedestrian_state"]["velocity"]
					idx += 1
				t_idx += 1
				self.sequence_idx[ii] += 1
				self.sequence_reset[ii] = 0

		return self.batch_enc, self.batch_dec, self.batch_dec_target, new_epoch

	def getBatch(self):
		"""
		Get the next batch of training data.
		"""
		# Update sequences
		# If batch sequences are filled and can be used or need to be updated.
		"""
				Get the next batch of training data.
		"""
		# If the dataset if smaller that the batch_size, repeat trajectories
		traj_id = 0
		while len(self.trajectory_set) < self.batch_size:
			self.trajectory_set.append(self.trajectory_set[traj_id])
			traj_id += 1
		# Update sequences
		# If batch sequences are empty and need to be filled
		if len(self.batch_sequences) == 0:
			for b in range(self.batch_size):
				trajectory = self.trajectory_set[self.data_idx]
				self.data_idx += 1
				self.batch_sequences.append(trajectory)

		# If batch sequences are filled and can be used or need to be updated.
		new_epoch = False
		for ii, traj in enumerate(self.batch_sequences):
			# if self.sequence_idx[ii] + self.prediction_horizon + self.tbpl + 1 >= len(traj):  # change trajectory
			if self.sequence_idx[ii] + self.prediction_horizon + self.tbpl >= len(traj):  # change trajectory
				self.data_idx = (self.data_idx + 1) % int(len(self.trajectory_set))
				if self.data_idx == 0:
					new_epoch = True
				self.sequence_idx[ii] = self.args.prev_horizon
				self.sequence_reset[ii] = 1
			else:
				self.sequence_reset[ii] = 0
		# fill the batch
		other_agents_pos = []

		for ii in range(self.batch_size):
			traj = self.batch_sequences[ii]
			start_idx = int(self.sequence_idx[ii])
			for tbp_step in range(self.tbpl):
				for prev_step in range(self.args.prev_horizon, -1, -1):
					robot_pos = traj[start_idx + tbp_step - prev_step]["robot_state"][:2]
					robot_vel = traj[start_idx + tbp_step - prev_step]["control_command"][:]
					ped_pos = traj[start_idx + tbp_step - prev_step]["pedestrian_state"]["position"]
					ped_vel = traj[start_idx + tbp_step - prev_step]["pedestrian_state"]["velocity"]

					self.batch_vel[ii, tbp_step, prev_step * self.args.input_state_dim:(prev_step + 1) * self.args.input_state_dim] = ped_vel
					self.batch_pos[ii, tbp_step, prev_step * self.args.input_state_dim:(prev_step + 1) * self.args.input_state_dim] = ped_pos

					self.batch_x[ii, tbp_step, prev_step * self.args.input_dim:(prev_step + 1) * self.args.input_dim] = np.array(
						[ped_pos[0], ped_pos[1], ped_vel[0], ped_vel[1]])

				# self.batch_goal[ii, tbp_step, :] = traj[start_idx + tbp_step]["pedestrian_goal_position"]
				self.batch_goal[ii, tbp_step, :] = traj[start_idx + tbp_step]["goal_position"]

				heading = math.atan2(ped_vel[1], ped_vel[0])

				# expands vector because the first axis represents an individual agent
				other_poses = np.expand_dims(robot_pos, axis=0)
				self.other_agents_pos[ii, tbp_step, :] = other_poses
				other_pos_local_frame = sup.positions_in_local_frame(ped_pos, heading, other_poses)
				radial_pedestrian_grid = sup.compute_radial_distance_vector(self.args.pedestrian_vector_dim,
				                                                            other_pos_local_frame,
				                                                            max_range=self.args.max_range_ped_grid,
				                                                            min_angle=0, max_angle=2 * np.pi,
				                                                            normalize=True)
				rel_pos = robot_pos - ped_pos
				rel_vel = robot_vel - ped_vel

				if (self.args.relative_info):
					if self.args.pedestrian_vector_dim == 3:
						self.other_agents_info[ii, tbp_step, :] = np.concatenate((np.array([np.linalg.norm(rel_pos)]), rel_vel))
					else:
						self.other_agents_info[ii, tbp_step, :] = np.concatenate((rel_pos, rel_vel))
				else:
					self.other_agents_info[ii, tbp_step, :] = radial_pedestrian_grid

				self.batch_grid[ii, tbp_step, :, :] = np.zeros((int(self.args.submap_width / self.args.submap_resolution),
				                                                int(self.args.submap_height / self.args.submap_resolution)))

				if self.args.agents_on_grid:
					self.add_other_agents_to_grid(self.batch_grid, ii, tbp_step, ped_pos, self.other_agents_pos)

				if self.args.rotated_grid:
					grid = sup.rotate_grid_around_center(grid, -heading * 180 / math.pi)  # rotation in degrees

				# Output values
				if self.args.predict_positions:
					data = "position"
				else:
					data = "velocity"
				for pred_step in range(self.prediction_horizon):
					if self.args.normalize_data:
						self.batch_target[ii, tbp_step, pred_step * self.args.output_dim] = \
							(traj[tbp_step + 1 + pred_step]["pedestrian_state"][data][0] - self.min_x) / (
									self.max_x - self.min_x)
						self.batch_target[ii, tbp_step, pred_step * self.args.output_dim + 1] = \
							(traj[tbp_step + 1 + pred_step]["pedestrian_state"][data][1] - self.min_y) / (
									self.max_y - self.min_y)
					else:
						self.batch_target[ii, tbp_step,
						pred_step * self.args.output_dim:pred_step * self.args.output_dim + self.args.output_dim] = \
							traj[start_idx + tbp_step + 1 + pred_step]["pedestrian_state"][data]

			self.sequence_idx[ii] += self.tbpl

		by = self.batch_target.copy()
		bx = self.batch_x.copy()

		# Rotate velocities into agent frame at each time step (x-axis is aligned with agents direction of motion)
		if self.args.rotated_grid:
			for batch_idx in range(self.batch_x.shape[0]):
				for tbp_step in range(self.batch_x.shape[1]):
					heading = math.atan2(bx[batch_idx, tbp_step, 3], bx[batch_idx, tbp_step, 2])
					rot_mat = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
					bx[batch_idx, tbp_step, 2:] = np.dot(rot_mat, bx[batch_idx, tbp_step, 2:])
					for pred_step in range(by.shape[2] / 2):
						by[batch_idx, tbp_step, 2 * pred_step:2 * pred_step + 2] = np.dot(rot_mat,
						                                                                  by[batch_idx, tbp_step,
						                                                                  2 * pred_step:2 * pred_step + 2])

		return bx, self.batch_vel, self.batch_pos, self.batch_goal, self.batch_grid, self.other_agents_info, by, self.other_agents_pos, new_epoch


	def add_other_agents_to_grid(self, batch_grid, seq_index, t_idx, ped_pos, other_agents):
		sigma = 3  # assumed position error
		rows = batch_grid.shape[2]
		cols = batch_grid.shape[3]
		# Todo expand this for more agents
		other_agents_pos = other_agents[seq_index][t_idx]
		for i in range(1):
			center_x = int(
				(other_agents_pos[0] - ped_pos[0] + rows / 2 * self.args.submap_resolution) / self.args.submap_resolution)
			center_y = int(
				(other_agents_pos[1] - ped_pos[1] + cols / 2 * self.args.submap_resolution) / self.args.submap_resolution)
			for idx in range(max(center_x - 5, 0), min(center_x + 5, rows)):
				for idy in range(max(center_y - 5, 0), min(center_y + 5, cols)):
					z = 1.0 * np.square(idx - center_x) / np.square(sigma) + 1.0 * np.square(
						idy - center_y) / np.square(
						sigma)
					pdf = max(0.0, 100.0 / 2.0 / np.pi / np.square(sigma) * np.exp(-z / 2.0))
					batch_grid[seq_index, t_idx, idx, idy] += pdf
					if batch_grid[seq_index, t_idx, idx, idy] > 1.0:
						batch_grid[seq_index, t_idx, idx, idy] = 1.0


	def getBatchTBP(self):
		"""
		Get the next batch of training data.
		"""
		self.batch_enc = np.zeros((self.batch_size, self.tbpl, self.args.input_dim * self.args.prev_horizon))
		self.batch_dec = np.zeros((self.batch_size, self.tbpl, self.args.output_dim))
		self.batch_dec_target = np.zeros((self.batch_size, self.tbpl, self.args.output_dim * self.args.prediction_horizon))
		# Update sequences
		# If batch sequences are empty and need to be filled
		trajectory = []
		if len(self.batch_sequences) == 0:
			for b in range(0, min(self.batch_size, len(self.trajectory_set) - len(self.trajectory_set) % self.batch_size)):
				trajectory = self.trajectory_set[self.data_idx]
				self.data_idx += 1
				self.batch_sequences.append(trajectory)
		# If batch sequences are filled and can be used or need to be updated.
		other_agents_pos = []
		new_epoch = False
		for ii, traj in enumerate(self.batch_sequences):
			if self.sequence_idx[ii] + self.tbpl + self.prediction_horizon + 1 >= len(traj):
				self.data_idx = (self.data_idx + 1) % int(len(self.trajectory_set))
				trajectory = self.trajectory_set[self.data_idx]
				if self.data_idx == 0:
					new_epoch = True
				self.batch_sequences[ii] = trajectory
				self.sequence_idx[ii] = self.prev_horizon
				self.sequence_reset[ii] = 1
			else:
				self.sequence_reset[ii] = 0

		# Fill the batch
		for ii in range(0, min(self.batch_size, len(self.trajectory_set) - len(
				self.trajectory_set) % self.batch_size)):  # min condition is used for small datasets
			traj = self.batch_sequences[ii]
			self.fillBatch(ii, int(self.sequence_idx[ii]), self.batch_enc, self.batch_dec, self.batch_dec_target, traj)
			self.sequence_idx[ii] += self.tbpl

		return self.batch_enc, self.batch_dec, self.batch_dec_target, new_epoch


	def fillBatch(self, batch_idx, start_idx, batch_enc, batch_dec, batch_dec_target, traj):
		"""
		Fill the data batches of batch_idx with data for all truncated backpropagation steps.
		"""
		other_agents_pos = []

		for tbp_step in range(self.tbpl):

			# Input values
			idx = 0
			for i in range(-self.prev_horizon, 0, 1):
				robot_pos = traj[start_idx + tbp_step + i]["robot_state"][0, :2]
				robot_vel = traj[start_idx + tbp_step + i]["control_command"][0, :]
				ped_pos = traj[start_idx + tbp_step + i]["pedestrian_state"]["position"]
				ped_vel = traj[start_idx + tbp_step + i]["pedestrian_state"]["velocity"]
				batch_enc[batch_idx, tbp_step, idx:idx + self.args.input_dim] = np.concatenate((robot_pos,
				                                                                                robot_vel,
				                                                                                ped_pos,
				                                                                                ped_vel))
				idx += self.args.input_dim

			ped_vel = traj[start_idx + tbp_step]["pedestrian_state"]["velocity"]

			radial_pedestrian_grid = sup.compute_radial_distance_vector(self.pedestrian_vector_dim,
			                                                            other_pos_local_frame,
			                                                            max_range=self.max_range_ped_grid, min_angle=0,
			                                                            max_angle=2 * np.pi,
			                                                            normalize=True)
			pedestrian_grid[batch_idx, tbp_step, :] = radial_pedestrian_grid

			batch_dec[batch_idx, tbp_step, :] = ped_vel
			idx = 0
			for i in range(0, self.prediction_horizon, 1):
				ped_vel = traj[start_idx + tbp_step + i]["pedestrian_state"]["velocity"]
				batch_dec_target[batch_idx, tbp_step, idx:idx + self.args.output_dim] = ped_vel
				idx += self.args.output_dim


	def getTrajectoryAsBatchTBP(self):
		"""
		Get the next batch of training data.
		"""
		id = np.random.randint(0, len(self.trajectory_set))
		traj = self.trajectory_set[id]
		start_index = 0 + self.prev_horizon
		end_index = len(traj) - self.prediction_horizon
		batch_enc_seq = []
		batch_dec_seq = []
		for t_idx in range(start_index, end_index):
			enc_seq = np.zeros((1, 8 * self.prev_horizon))
			dec_seq = np.zeros((1, 2))
			idx = 0
			for i in range(t_idx - self.prev_horizon, t_idx, 1):
				robot_pos = traj[i]["robot_state"][0, :2]
				robot_vel = traj[i]["control_command"][0, :]
				ped_pos = traj[i]["pedestrian_state"]["position"]
				ped_vel = traj[i]["pedestrian_state"]["velocity"]
				input_data = np.concatenate((robot_pos, robot_vel, ped_pos, ped_vel))
				enc_seq[0, idx:idx + 8] = input_data
				idx += 8
			idx = 0
			batch_enc_seq.append(enc_seq)

			dec_seq[0, :] = traj[t_idx]["pedestrian_state"]["velocity"]

			batch_dec_seq.append(dec_seq)

		return batch_enc_seq, batch_dec_seq, traj


	def getTrajectoryAsBatch(self, trajectory_idx, max_sequence_length=1000):
		"""
		Get a trajectory out of the trajectory set in the same format as for the standard training data
		(e.g. for validation purposes).
		"""
		traj = self.trajectory_set[trajectory_idx]

		sequence_length = min(max_sequence_length, len(traj) - self.prediction_horizon)
		batch_x = np.zeros([1, sequence_length, self.args.input_dim*(self.prev_horizon+1)])  # data fed for training
		batch_pos = np.zeros([1, sequence_length, self.args.input_state_dim*(self.prev_horizon+1)])  # data fed for training
		batch_vel = np.zeros([1, sequence_length, self.args.input_state_dim*(self.prev_horizon+1)])  # data fed for training
		batch_goal = np.zeros([1, sequence_length, 2])
		batch_target = np.zeros([1, sequence_length, self.args.output_dim * self.prediction_horizon])

		other_agents_info = np.zeros([1, sequence_length, self.args.pedestrian_vector_dim])
		batch_grid = np.zeros((1, sequence_length, int(self.args.submap_width / self.args.submap_resolution),
		                       int(self.args.submap_height / self.args.submap_resolution)))
		other_agents_pos = np.zeros((1, sequence_length, self.args.pedestrian_vector_state_dim))
		# fill the batch
		start_idx = self.args.prev_horizon
		for tbp_step in range(sequence_length):
			for prev_step in range(self.args.prev_horizon, -1, -1):
				robot_pos = traj[start_idx + tbp_step - prev_step]["robot_state"][:2]
				robot_vel = traj[start_idx + tbp_step - prev_step]["control_command"][:]
				ped_pos = traj[start_idx + tbp_step - prev_step]["pedestrian_state"]["position"]
				ped_vel = traj[start_idx + tbp_step - prev_step]["pedestrian_state"]["velocity"]

				batch_vel[0, tbp_step,
				prev_step * self.args.input_state_dim:(prev_step + 1) * self.args.input_state_dim] = ped_vel
				batch_pos[0, tbp_step,
				prev_step * self.args.input_state_dim:(prev_step + 1) * self.args.input_state_dim] = ped_pos

				batch_x[0, tbp_step, prev_step * self.args.input_dim:(prev_step + 1) * self.args.input_dim] = np.array(
					[ped_pos[0], ped_pos[1], ped_vel[0], ped_vel[1]])

			# batch_goal[0, tbp_step, :] = traj[tbp_step]["pedestrian_goal_position"]
			batch_goal[0, tbp_step, :] = traj[tbp_step]["goal_position"]

			heading = math.atan2(robot_vel[1], robot_vel[0])

			# expands vector because the first axis represents an individual agent
			other_poses = robot_pos
			other_agents_pos[0, tbp_step, :] = other_poses
			other_pos_local_frame = sup.positions_in_local_frame(ped_pos, heading, np.expand_dims(other_poses, axis=0))
			radial_pedestrian_grid = sup.compute_radial_distance_vector(self.args.pedestrian_vector_dim,
			                                                            other_pos_local_frame,
			                                                            max_range=self.args.max_range_ped_grid,
			                                                            min_angle=0, max_angle=2 * np.pi,
			                                                            normalize=True)
			rel_pos = robot_pos - ped_pos
			rel_vel = robot_vel - ped_vel

			if (self.args.relative_info):
				if self.args.pedestrian_vector_dim == 3:
					other_agents_info[0, tbp_step, :] = np.concatenate((np.array([np.linalg.norm(rel_pos)]), rel_vel))
				else:
					other_agents_info[0, tbp_step, :] = np.concatenate((rel_pos, rel_vel))
			else:
				other_agents_info[0, tbp_step, :] = radial_pedestrian_grid

			# self.add_other_agents_to_grid(batch_grid,0, tbp_step, ped_pos, other_agents_pos)

			# Output values
			for pred_step in range(self.prediction_horizon):
				if self.args.normalize_data:
					batch_target[0, tbp_step, pred_step * self.args.output_dim] = \
						(traj[tbp_step + 1 + pred_step]["pedestrian_state"]["velocity"][0] - self.min_x) / (self.max_x - self.min_x)
					batch_target[0, tbp_step, pred_step * self.args.output_dim + 1] = \
						(traj[tbp_step + 1 + pred_step]["pedestrian_state"]["velocity"][1] - self.min_y) / (self.max_y - self.min_y)
				else:
					batch_target[0, tbp_step,
					pred_step * self.args.output_dim:pred_step * self.args.output_dim + self.args.output_dim] = \
						traj[tbp_step + 1 + pred_step]["pedestrian_state"]["velocity"]

		return batch_x, batch_vel, batch_pos, batch_goal, batch_grid, other_agents_info, batch_target, other_agents_pos, traj


	def getTrajectoryAsBatchSeqToSeq(self):
		"""
		Get the next batch of training data.
		"""
		id = np.random.randint(0, len(self.trajectory_set))
		traj = self.trajectory_set[id]
		start_index = 0 + self.prev_horizon
		end_index = len(traj) - self.prediction_horizon
		batch_enc_seq = []
		batch_dec_seq = []
		for t_idx in range(start_index, end_index):
			enc_seq = np.zeros((self.prev_horizon, 8))
			dec_seq = np.zeros((self.prediction_horizon, 2))
			idx = 0
			for i in range(t_idx - self.prev_horizon, t_idx, 1):
				robot_pos = traj[i]["robot_state"][:2]
				robot_vel = traj[i]["control_command"][:]
				ped_pos = traj[i]["pedestrian_state"]["position"]
				ped_vel = traj[i]["pedestrian_state"]["velocity"]
				input_data = np.concatenate((robot_pos, robot_vel, ped_pos, ped_vel))
				enc_seq[idx, :] = input_data
				idx += 1
			idx = 0
			batch_enc_seq.append(enc_seq)
			for i in range(t_idx, t_idx + self.prediction_horizon, 1):
				dec_seq[idx, :] = traj[i]["pedestrian_state"]["velocity"]
				idx += 1
			batch_dec_seq.append(dec_seq)

		return batch_enc_seq, batch_dec_seq, traj


	def calc_scale(self, keep_ratio=True):
		self.sx = 1 / (self.max_x - self.min_x)
		self.sy = 1 / (self.max_y - self.min_y)
		if keep_ratio:
			if self.sx > self.sy:
				self.sx = self.sy
			else:
				self.sy = self.sx


	def normalize(self, data, shift=True, inPlace=True):
		if inPlace:
			data_copy = data
		else:
			data_copy = np.copy(data)

		if data.ndim == 1:
			data_copy[0] = (data[0] - self.min_x * shift) * self.sx
			data_copy[1] = (data[1] - self.min_y * shift) * self.sy
		elif data.ndim == 2:
			data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
			data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
		elif data.ndim == 3:
			data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
			data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
		elif data.ndim == 4:
			data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_x * shift) * self.sx
			data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_y * shift) * self.sy
		else:
			return False
		return data_copy


	def denormalize(self, data, shift=True, inPlace=False):
		if inPlace:
			data_copy = data
		else:
			data_copy = np.copy(data)

		ndim = data.ndim
		if ndim == 1:
			data_copy[0] = data[0] / self.sx + self.min_x * shift
			data_copy[1] = data[1] / self.sy + self.min_y * shift
		elif ndim == 2:
			data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
			data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
		elif ndim == 3:
			data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
			data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
		elif ndim == 4:
			data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx + self.min_x * shift
			data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy + self.min_y * shift
		else:
			return False

		return data_copy


	def save_data(self):
		file = open(self.args.data_path, 'wb')
		print("Saving data to: '{}'".format(self.args.data_path))
		# self.dataset.pop(0)
		pkl.dump(self.trajectory_set, file)

		file.close()


	def load_data(self):
		print("Loading data from: '{}'".format(self.args.data_path + self.args.dataset))
		self.file = open(self.args.data_path + self.args.dataset, 'rb')
		if sys.version_info[0] < 3:
			tmp_self = pkl.load(self.file)
		else:
			tmp_self = pkl.load(self.file, encoding='latin1')
		self.trajectory_set = tmp_self
		trajectory_set = []
		# Get normalization constants
		self.pedestrian_dataset = []
		self.robot_dataset = []
		self.max_x = -100
		self.min_x = 100
		self.max_y = -100
		self.min_y = 100
		for traj_id in range(len(self.trajectory_set)):
			traj = self.trajectory_set[traj_id]
			self.robot_vel = np.zeros((len(traj), 2))
			self.pedestrian_vel = np.zeros((len(traj), 2))
			sub_traj = []
			# Sub-sample trajectories to make dt = 0.4
			for t_id in range(0, len(traj), 4):
				self.robot_vel[t_id, :] = traj[t_id]["control_command"]
				self.pedestrian_vel[t_id, :] = traj[t_id]["pedestrian_state"][""]
				self.max_x = max(self.max_x, traj[t_id]["pedestrian_state"]["position"][0])
				self.min_x = min(self.min_x, traj[t_id]["pedestrian_state"]["position"][0])
				self.max_y = max(self.max_y, traj[t_id]["pedestrian_state"]["position"][1])
				self.min_y = min(self.min_y, traj[t_id]["pedestrian_state"]["position"][1])
				sub_traj.append(traj[t_id])
			# Append zero velocities to goal position
			if traj:
				goal_data = traj[-1].copy()
				goal_data["pedestrian_state"]["velocity"] = np.array([0, 0])
				for t_id in range(10):
					sub_traj.append(goal_data)
				if len(sub_traj) > self.min_length_trajectory:
					trajectory_set.append(sub_traj)
				self.robot_dataset.append(self.robot_vel)
				self.pedestrian_dataset.append(self.pedestrian_vel)
		self.robot_trajectory_set = np.asarray(self.robot_dataset)
		self.pedestrian_dataset = np.asarray(self.pedestrian_dataset)
		self.trajectory_set = trajectory_set
		self.calc_scale()
		# random.shuffle(self.trajectory_set)


	def load_map(self):
		try:
			file_path = self.args.data_path + '/map.npy'
			grid_map = np.load(file_path)[()]
			print("Map loaded")
			return grid_map
		except IOError:
			print("ERROR: Cannot load map.")
			return False


	def start_node(self):
		rospy.init_node('OnlineDataHandler', anonymous=True)
		# start node to collect data online
		while not rospy.is_shutdown():  # while the node is running, get the complete data, input and output batches
			self.collect_data()
			if len(self.trajectory_set) > self.args.max_trajectories:
				break
			sleep(1.0 / self.rate)
		self.save_data()


if __name__ == '__main__':
	data_prep = DataHandler()
	data_prep.start_node()
