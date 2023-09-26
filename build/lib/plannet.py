#!/usr/bin/env python
import time
import rospy
import rospkg
import cv2
from geometry_msgs.msg import Pose, PoseStamped
from pedsim_msgs.msg import TrackedPersons, TrackedPerson
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
import tf as transferfunction
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/media/bdebrito/7697ec91-468a-4763-b1c3-135caa7f5aed/home/code/I-LSTM/src/data_utils'])
sys.path.extend(['/media/bdebrito/7697ec91-468a-4763-b1c3-135caa7f5aed/home/code/I-LSTM/src/models'])
import os
import math
import json
if sys.version_info[0] < 3:
	import Support as sup
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils import Support as sup

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
import threading as th
from lmpcc_msgs.msg import lmpcc_obstacle, lmpcc_obstacle_array, lmpcc_predict_point, lmpcc_predict_path
import pickle as pkl


model_name = "Plan_VGDNN"

# Import model
from RNN_plannet import NetworkModel

class PlanNet:
	def __init__(self):

		self.model_name = rospy.get_param('~model_name', 'RNN_plannet')
		self.model_id = rospy.get_param('~id', '8')
		self.robot_state_topic = rospy.get_param('~robot_state_topic', '/robot_state')
		self.goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
		self.other_agents_topic = rospy.get_param('~other_agents_topic',"/pedsim_visualizer/tracked_persons")
		self.grid_topic = rospy.get_param('~grid_topic',"/move_base/local_costmap/costmap")

		# Load Model Parameters
		self.load_args()

		# State variables
		self.current_position_ = np.zeros([1,1,(self.model_args.prev_horizon+1)*2])
		self.current_velocity_ = np.zeros([1,1,(self.model_args.prev_horizon+1)*2])
		self.goal_ = np.zeros([1,1,2])
		self.rel_goal = np.zeros([1,1,self.model_args.goal_size])
		self.n_other_agents = 8
		self.other_agents = np.ones([1,1,self.model_args.pedestrian_vector_dim])
		self.other_agents_info = np.zeros([1, 1, self.model_args.pedestrian_vector_dim])

		self.n_peds = 8

		self.other_pedestrians = []
		for i in range(self.n_peds):
			self.other_pedestrians.append(TrackedPerson())

		self.load_model()

		self.width = int(self.model_args.submap_width / self.model_args.submap_resolution )
		self.height = int(self.model_args.submap_height / self.model_args.submap_resolution)
		self.grid = np.zeros([1, 1, self.width, self.height])
		self.fig_animate = pl.figure('Animation')
		self.fig_width = 12  # width in inches
		self.fig_height = 25  # height in inches
		self.fig_size = [self.fig_width, self.fig_height]
		self.fontsize = 9
		self.colors = ['r', 'g', 'y']
		self.params = {'backend': 'ps',
		          'axes.labelsize': self.fontsize,
		          'font.size': self.fontsize,
		          'xtick.labelsize': self.fontsize,
		          'ytick.labelsize': self.fontsize,
		          'figure.figsize': self.fig_size}
		pl.rcParams.update(self.params)
		#self.ax_pos = pl.subplot()
		pl.show(block=False)

		# ROS Subscribers
		rospy.Subscriber(self.robot_state_topic, PoseStamped, self.robot_state_CB, queue_size=1)
		rospy.Subscriber(self.goal_topic,PoseStamped,self.goal_CB, queue_size=1)
		rospy.Subscriber(self.other_agents_topic, TrackedPersons, self.other_agents_CB, queue_size=1)
		rospy.Subscriber(self.grid_topic, OccupancyGrid, self.grid_CB, queue_size=1)

		# ROS Publishers
		self.tf_listener = transferfunction.TransformListener()
		self.tf_transformer = transferfunction.TransformerROS()

		self.pub_viz = rospy.Publisher('plannet_trajectory', Path, queue_size=10)
		self.pub_static_collision_traj = rospy.Publisher('nn_predicted_trajectory', lmpcc_predict_path, queue_size=10)
		self.pub_goal_marker = rospy.Publisher('goal_marker', Marker, queue_size=1)

		# THread control
		self.lock = th.Lock()

	def load_args(self):
		cwd = os.getcwd()

		model_path = os.path.normpath(cwd + '/../') + '/trained_models/' + self.model_name + "/" + str(self.model_id)

		print("Loading data from: '{}'".format(model_path))
		file = open(model_path + '/model_parameters.pkl', 'rb')
		if sys.version_info[0] < 3:
			model_parameters = pkl.load(file)  # ,encoding='latin1')
		else:
			model_parameters = pkl.load(file, encoding='latin1')
		file.close()
		self.model_args = model_parameters["args"]

		if not(os.path.isfile(model_path + '/model_parameters.json')):
			json.dump(vars(self.model_args),open(model_path + '/model_parameters.json','w'))

		# change some args because we are doing inference
		self.model_args.truncated_backprop_length = 1
		self.model_args.batch_size = 1

	def load_model(self):

		self.model = NetworkModel(self.model_args)

		self.sess = tf.Session()

		self.model.warmstart_model(self.model_args, self.sess)
		try:
			self.model.warmstart_convnet(self.model_args, self.sess)
		except:
			print("No convnet")

		print("Model Initialized")

	def robot_state_CB(self, data):
		# Robot state callback
		# shift
		self.current_velocity_ = np.roll(self.current_velocity_, 2, axis=2)
		self.current_position_ = np.roll(self.current_position_, 2, axis=2)

		self.current_position_[0,0,0] = data.pose.position.x
		self.current_position_[0,0,1] = data.pose.position.y
		orientation = data.pose.orientation.z
		linear_velocity = data.pose.position.z

		self.current_velocity_[0,0,0] = linear_velocity*np.cos(orientation)
		self.current_velocity_[0,0,1] = linear_velocity*np.sin(orientation)

	def goal_CB(self, data):
		# Goal Callback
		self.goal_[0,0,0] = data.pose.position.x
		self.goal_[0,0,1] = data.pose.position.y

		rel_pos = self.goal_ - self.current_position_[0,0,:2]

		for t in range(self.model_args.prev_horizon+1):
			self.current_velocity_[0, 0, 2*t] = np.tanh(rel_pos[0,0,0]/self.model_args.dt)
			self.current_velocity_[0, 0, 2*t+1] = np.tanh(rel_pos[0,0,1]/self.model_args.dt)

		self.publish_goal_marker()

	def grid_CB(self, data):
		# scale the grid from 0-100 to 0-1 and invert
		print("Grid data size: " + str(len(data.data)))
		self.grid[0,0,:,:] = (np.asarray(data.data).reshape((self.width, self.height)).astype(float) / 100.0)
		self.grid[0,0,:,:] = np.flip(self.grid[0,0,:,:], 1)
		self.grid[0,0,:,:] = sup.rotate_grid_around_center(self.grid[0,0,:,:], 90)

		if False:
			self.ax_pos.clear()
			sup.plot_grid(self.ax_pos, np.array([0.0, 0.0]), self.grid, self.model_args.submap_resolution,
			              np.array([self.model_args.submap_width, self.model_args.submap_height]))
			self.ax_pos.set_xlim([-self.model_args.submap_width/2,self.model_args.submap_width/2])
			self.ax_pos.set_ylim([-self.model_args.submap_height/2,self.model_args.submap_height/2])
			self.ax_pos.set_aspect('equal')

	def other_agents_CB(self, data):
		if len(data.tracks) != self.n_peds:
			print("NUmber of peds do not match callback info")

		for person_it in range(self.n_peds):
			ped = TrackedPerson()
			ped.pose=data.tracks[person_it].pose
			ped.track_id = data.tracks[person_it].track_id
			ped.twist = data.tracks[person_it].twist
			self.other_pedestrians[person_it] = ped

	def fillBatchOtherAgents(self):

		other_poses_ordered = np.zeros((len(self.other_pedestrians), 6))

		for ag_id in range(len(self.other_pedestrians)):
			other_poses_ordered[ag_id, 0] = self.other_pedestrians[ag_id].pose.pose.position.x
			other_poses_ordered[ag_id, 1] = self.other_pedestrians[ag_id].pose.pose.position.y
			other_poses_ordered[ag_id, 2] = self.other_pedestrians[ag_id].twist.twist.linear.x
			other_poses_ordered[ag_id, 3] = self.other_pedestrians[ag_id].twist.twist.linear.y
			other_poses_ordered[ag_id, 4] = np.linalg.norm(other_poses_ordered[ag_id, :2] - self.current_position_[0,0,:2])
			other_poses_ordered[ag_id, 5] = self.other_pedestrians[ag_id].track_id

		#other_poses_ordered = other_poses_ordered[other_poses_ordered[:, 4].argsort()]
		heading = math.atan2(self.current_velocity_[0,0,1], self.current_velocity_[0,0,0])
		other_pos_local_frame = sup.positions_in_local_frame(self.current_position_[0,0,:2], heading, other_poses_ordered[:,:2])
		self.other_agents_info[0,0] = sup.compute_radial_distance_vector(self.model_args.pedestrian_vector_dim, other_pos_local_frame,
		                                                            max_range=self.model_args.max_range_ped_grid, min_angle=0,
		                                                            max_angle=2 * np.pi,
		                                                            normalize=True)

	# query feed the data into the net and calculates the trajectory
	def query(self):
		# save current position and orientation to calculate the prediction from

		self.fillBatchOtherAgents()

		rel_goal = self.goal_[0,0,:2] -self.current_position_[0,0,:2]

		#self.rel_goal[0,0, 0] = np.linalg.norm(rel_goal)
		#self.rel_goal[0,0, 1] = np.arctan2(rel_goal[1], rel_goal[0])
		self.rel_goal[0,0, :] = rel_goal/np.linalg.norm(rel_goal)

		dict = {"batch_vel": self.current_velocity_,
		        "batch_ped_grid": self.other_agents_info,
		        "batch_goal": self.rel_goal,
		        "step": 0
		        }
		feed_dict_ = self.model.feed_test_dic(**dict)

		outputs = self.model.predict(self.sess, feed_dict_, True)

		# publish the predicted trajectories
		local_trajectory, global_trajectory = self.calculate_trajectory(outputs[0])
		self.local_trajectory = local_trajectory
		self.global_trajectory = global_trajectory

	# add the velocity predictions together to form points and convert them to global coordinate frame
	def calculate_trajectory(self, y_model_pred):
		local_trajectory = np.empty([self.model_args.prediction_horizon, 2])
		global_trajectory = Path()
		global_trajectory.header.frame_id = "odom"
		global_trajectory.header.stamp = rospy.Time.now()
		x = 0.0
		y = 0.0
		x_global = self.current_position_[0,0,0]
		y_global = self.current_position_[0,0,1]
		mix_idx = 0
		for pred_step in range(self.model_args.prediction_horizon):
			pose = PoseStamped()
			idx = pred_step * self.model_args.output_pred_state_dim * self.model_args.n_mixtures + mix_idx
			x = x + y_model_pred[0][0][idx] * self.model_args.dt
			y = y + y_model_pred[0][0][idx + self.model_args.n_mixtures] * self.model_args.dt
			local_trajectory[pred_step, :] = np.array([x, y])
			x_global += y_model_pred[0][0][idx] * self.model_args.dt
			y_global += y_model_pred[0][0][idx + self.model_args.n_mixtures] * self.model_args.dt

			pose.pose.position.x = x_global
			pose.pose.position.y = y_global
			pose.pose.orientation.w = 1.0
			global_trajectory.poses.append(pose)

		return local_trajectory, global_trajectory

	# gather the trajectories from the different pedestrians and publish their visualisation
	def publish_trajectory_visualisation(self):

		self.pub_viz.publish(self.global_trajectory)


	def publish_goal_marker(self):
		marker = Marker()
		marker.header.frame_id = "odom"
		marker.header.stamp = rospy.Time.now()
		marker.ns = "goal_marker"
		marker.id = 0
		marker.type = 3
		marker.pose.position.z = 0.1
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = 1.2
		marker.scale.y = 1.2
		marker.scale.z = 1.1
		marker.pose.position.x = self.goal_[0,0,0]
		marker.pose.position.y = self.goal_[0,0,1]
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 0.0
		marker.color.b = 0.0
		marker.lifetime = rospy.rostime.Duration()
		self.pub_goal_marker.publish(marker)


def visualize_ped_grid(self, ped_grid):
	pedestrian_vector_dim = 72
	max_range_ped_grid = 1.0
	fig1 = plt.figure("Angular grid")
	ax_ped_grid = plt.subplot()
	ax_ped_grid.clear()
	grid = ped_grid[0]
	grid = np.expand_dims(grid, axis=1)
	grid_flipped = np.zeros_like(grid)
	grid_flipped[0:int(pedestrian_vector_dim / 2)] = grid[-int(pedestrian_vector_dim / 2):]
	grid_flipped[int(pedestrian_vector_dim / 2):] = grid[0:int(pedestrian_vector_dim / 2)]
	sup.plot_radial_distance_vector(ax_ped_grid, grid_flipped, max_range=1.0, min_angle=0.0,
	                                max_angle=2 * np.pi)
	ax_ped_grid.plot(30, 30, color='r', marker='o', markersize=4)
	ax_ped_grid.arrow(0, 0, 1, 0, head_width=0.1, head_length=max_range_ped_grid)  # agent poiting direction
	# x- and y-range only need to be [-1, 1] since the pedestrian grid is normalized
	ax_ped_grid.set_xlim([-max_range_ped_grid - 1, max_range_ped_grid + 1])
	ax_ped_grid.set_ylim([-max_range_ped_grid - 1, max_range_ped_grid + 1])

	fig1.canvas.draw()
	plt.show(block=False)


if __name__ == '__main__':
	rospy.init_node('PlanNet_node')
	planning_network = PlanNet()
	rospy.sleep(5.0)
	while not rospy.is_shutdown():
		start_time = time.time()

		planning_network.lock.acquire()
		planning_network.query()
		planning_network.publish_trajectory_visualisation()
		planning_network.lock.release()
		#planning_network.fig_animate.canvas.draw()
		#cv2.imshow("image", planning_network.grid)
		#cv2.waitKey(100)
		# wait around a bit if neccesairy
		now = time.time()
		if now - start_time < 0.05:  # args.dt:
			rospy.sleep(0.05 - (now - start_time))
		else:
			print("not keeping up to rate")
	planning_network.sess.close()
