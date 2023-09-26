#!/usr/bin/env python3

import os, sys; print("Running with {}".format(sys.version))
sys.path.append('../')
import copy, time, pickle, math, collections

from scipy.spatial.transform import Rotation
import tensorflow as tf
import numpy as np; np.set_printoptions(suppress=True)
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

from models.VGDNN import NetworkModel as SocialVRNN
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import rospy, rospkg
import message_filters
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array, svrnn_path as SVRNN_Path, svrnn_path_array as SVRNN_Path_Array
from data_utils.OccupancyGrid import OccupancyGrid as OccGrid

#sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2

PACKAGE_NAME = 'social_vrnn'
MAX_QUERY_AGENTS = 3
SWITCH_AXES = False


class SVRNN_Predictor():
    
    def __init__(self, node_name, visual_node_name):
        self.node_name, self.visual_node_name = node_name, visual_node_name

        # Bind node
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.shutdown_callback)
        rospy.loginfo('{} has started.'.format(self.visual_node_name))

        # Set up class variables
        self.get_submap = None # function to request submap for some position
        self.agents_pos, self.agents_vel = {}, {} # dictionary of world state

        # Generic marker template
        mrk = Marker()
        mrk.header.frame_id = 'odom'
        mrk.action = Marker.ADD
        mrk.lifetime.secs = int(1.0)
        mrk.scale.x, mrk.scale.y, mrk.scale.z = 1.0, 1.0, 1.0
        mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = 1.0, 1.0, 1.0, 1.0
        mrk.pose.orientation.w = 1.0
        self.marker_template = mrk

        # Read parameters
        self.n_agents = rospy.get_param("/n_agents")

        # Initialize map class
        self.occgrid = OccGrid()

        # Load the model
        self.model_args = self.get_model_args('VGDNN', '33')
        self.model, self.tf_session = self.load_model(SocialVRNN, self.model_args)

        # Set up subscribers
        rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self.store_occupancy_grid)

        # Set up publishers
        self.pred_path_publisher = rospy.Publisher('/{}/predictions'.format(self.node_name), SVRNN_Path_Array, latch=True, queue_size=10)
        self.pos_mark_publisher = rospy.Publisher('/{}/markers/positions'.format(self.node_name), MarkerArray, latch=True, queue_size=10)
        self.pred_mark_publisher = rospy.Publisher('/{}/markers/predictions'.format(self.node_name), MarkerArray, latch=True, queue_size=10)


    def infer(self):
        if self.get_submap is None: return None, None
        #print("Agent pos: ", self.agents_pos)
        if len(self.agents_pos) < 2:
            rospy.logwarn('No other agents present, skipping path prediction')
            return None, None

        # Build NumPy matrices from world state
        active_agents = np.full((MAX_QUERY_AGENTS, ), False, dtype=bool)
        for id in self.agents_pos.keys(): active_agents[id] = True

        agents_pos_np = np.zeros((MAX_QUERY_AGENTS, 2), dtype=float)
        for id, pos in self.agents_pos.items(): agents_pos_np[id] = pos

        agents_vel_np = np.zeros((MAX_QUERY_AGENTS, 2), dtype=float)
        for id, vel in self.agents_vel.items(): agents_vel_np[id] = vel
        
        # define multiplier based on dt. (dt = 0.4 becomes 4 since incoming time steps are recorded 0.1 seconds from each other)
        dt = int(self.model_args.dt * 10)

        # Maintain past agents velocities
        if not hasattr(self, '_agents_vel_ls_all'):
            self._agents_vel_ls_all = np.zeros((MAX_QUERY_AGENTS, 2 * (self.model_args.prev_horizon + 1) * dt), dtype=float)
        self._agents_vel_ls_all = np.roll(self._agents_vel_ls_all, 2, axis=1) # Brings oldest velocities to front of array
        self._agents_vel_ls_all[:, :2] = agents_vel_np # Replaces oldest velocities with newest

        # Only use time steps at dt interval
        if not hasattr(self, '_agents_vel_ls'):
            self._agents_vel_ls = np.zeros((MAX_QUERY_AGENTS, 2 * (self.model_args.prev_horizon + 1)), dtype=float)
        self._agents_vel_ls[:, ::2] = self._agents_vel_ls_all[:, ::(dt*2)]
        self._agents_vel_ls[:, 1::2] = self._agents_vel_ls_all[:, 1::(dt*2)]

        # Calculate the relative odometry
        relative_odometry = np.zeros((MAX_QUERY_AGENTS, 4 * self.model_args.n_other_agents), dtype=float)
        for sub_ind in np.arange(MAX_QUERY_AGENTS)[active_agents]:

            # Get n-nearest neighbours
            nn_inds = np.argsort(np.linalg.norm(agents_pos_np[active_agents] - agents_pos_np[sub_ind], axis=1))[1:]
            #if len(nn_inds) < self.model_args.n_other_agents: nn_inds = np.concatenate((nn_inds, np.repeat(nn_inds[-1], self.model_args.n_other_agents - len(nn_inds))))
            #if len(nn_inds) > self.model_args.n_other_agents: nn_inds = nn_inds[:self.model_args.n_other_agents]

            # Calculate relative positions
            nn_pos = agents_pos_np[active_agents][nn_inds].copy()
            #multivars = []
            #for i in range(len(nn_inds)):
            #    multivars.append( multivariate_normal.pdf(  np.linalg.norm(nn_pos - agents_pos_np[sub_ind], axis=1)[i] , mean=0.0, cov=5.0) )
            nn_pos -= agents_pos_np[sub_ind] 
            
            #for i in range(len(nn_inds)):
            #    nn_pos[i] = nn_pos[i] * multivars[i]

            # Calculate relative velocities
            nn_vel = agents_vel_np[active_agents][nn_inds].copy()
            nn_vel -= agents_vel_np[sub_ind]

            #relative_odometry[sub_ind] = np.concatenate((nn_pos, nn_vel,np.linalg.norm(nn_pos),np.arctan2(nn_pos[1], nn_pos[0])), axis=1).flatten()
            #relative_odometry[sub_ind, sub_ind*4:sub_ind*4+((self.n_agents-1)*4)] = np.concatenate([nn_pos, nn_vel], axis=1).flatten()
            relative_odometry[sub_ind, 0:((self.n_agents-1)*4)] = np.concatenate([nn_pos, nn_vel], axis=1).flatten()

        # Get the submaps
        submaps = np.zeros((MAX_QUERY_AGENTS, int(self.model_args.submap_width / self.model_args.submap_resolution), int(self.model_args.submap_height / self.model_args.submap_resolution))) #, dtype=float
        for id in np.arange(MAX_QUERY_AGENTS)[active_agents]:
            if self.get_submap is None: return
            submaps[id] = self.get_submap(agents_pos_np[id], agents_vel_np[id])

        
        #print("Batch_vel : ", self._agents_vel_ls)
        """
        print("batch vel shape: ", self._agents_vel_ls.shape[0], self._agents_vel_ls.shape[1]) 

        print("Relative vector: ", relative_odometry)
        print("batch rel shape: ", relative_odometry.shape[0], relative_odometry.shape[1]) 
        """

        # Predict the future velocities
        return copy.deepcopy(self.agents_pos), self.model.predict(
            self.tf_session,
            self.model.feed_pred_dic(
                batch_vel = self._agents_vel_ls,
                batch_ped_grid = relative_odometry,
                batch_grid = submaps,
                step = 0
            ),
            True
        )[0]

    def rotate_to_world_frame(self, msg):

        # Read current position from message in world frame
        curr_x, curr_y = msg.pose.pose.position.x, msg.pose.pose.position.y
        
        # Read current velocity from message in body frame of roboat
        curr_vx, curr_vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y

        # Transfrom velocity to world frame
        q0, q1, q2, q3 = (msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w)
        
        rot_mat = R.from_quat([q0, q1, q2, q3])
        curr_v = np.array([curr_vx, curr_vy, 0])

        # Transformation
        v = rot_mat.apply(curr_v)

        return curr_x, curr_y, v[0], v[1]

    def publish(self, agents_pos, pred_velocities):
        if pred_velocities is None: return

        # Publish predicted positions
        pred_velocities_path = SVRNN_Path_Array()
        pred_velocities_mrk = MarkerArray()
        for id in agents_pos.keys():
            pred = pred_velocities[id]
            path_msg = SVRNN_Path()
            path_msg.id = float(id)
            path_msg.dt = self.model_args.dt
            for mx_id in range(self.model_args.n_mixtures):
                main_mixture = mx_id == 0
                path_mrk = copy.deepcopy(self.marker_template)
                path_mrk.id = self.model_args.n_mixtures * id + mx_id
                path_mrk.type = Marker.LINE_STRIP
                path_mrk.scale.x = 0.1
                path_mrk.pose.position.x = agents_pos[id][1] if SWITCH_AXES else agents_pos[id][0]
                path_mrk.pose.position.y = agents_pos[id][0] if SWITCH_AXES else agents_pos[id][1]
                path_mrk.color.r = float(mx_id == 0)
                path_mrk.color.g = float(mx_id == 1)
                path_mrk.color.b = float(mx_id == 2)
                prev_x, prev_y = 0.0, 0.0
                for ts_id in range(self.model_args.prediction_horizon):
                    pose = PoseStamped()
                    pt = Point()
                    idx = ts_id * self.model_args.output_pred_state_dim * self.model_args.n_mixtures + mx_id
                    pt.x = prev_x + self.model_args.dt * pred[0][idx + (self.model_args.n_mixtures if SWITCH_AXES else 0)]
                    pt.y = prev_y + self.model_args.dt * pred[0][idx + (0 if SWITCH_AXES else self.model_args.n_mixtures)]
                    pose.pose.position = pt
                    if main_mixture: path_msg.path.poses.append(pose)
                    pt.z = 0.2
                    path_mrk.points.append(pt)
                    prev_x, prev_y = pt.x, pt.y
                pred_velocities_mrk.markers.append(path_mrk)
            pred_velocities_path.paths.append(path_msg)
        #print("Pred positions", pred_velocities_path)
        self.pred_path_publisher.publish(pred_velocities_path)
        self.pred_mark_publisher.publish(pred_velocities_mrk)


    def store_occupancy_grid(self, occupancy_grid_msg):
        # Reset class state
        def safedelattr(attr):
            if hasattr(self, attr): delattr(self, attr)
        safedelattr('_roboat_pos'); safedelattr('_roboat_last_update')
        self.agents_pos, self.agents_vel = {}, {}; safedelattr('_agents_last_update')
        self.get_submap = None; safedelattr('_agents_vel_ls')
        rospy.loginfo('Cleared history context')

        # Makes 1 pixel equal to 1 submap pixel
        scale_factor = occupancy_grid_msg.info.resolution / self.model_args.submap_resolution

        # Transform occupancy grid data into Social-VRNN format
        grid_info = occupancy_grid_msg.info
        occupancy_grid = np.asarray(occupancy_grid_msg.data, dtype=float).reshape((grid_info.height, grid_info.width))
        occupancy_grid[occupancy_grid > 0.0] = 1.0
        occupancy_grid[occupancy_grid < 1.0] = 0.0
        occupancy_grid = cv2.flip(occupancy_grid, 0)
        occupancy_grid = cv2.resize(
            occupancy_grid,
            (int(scale_factor * grid_info.width), int(scale_factor * grid_info.height)),
            fx=0, fy=0,
            interpolation=cv2.INTER_NEAREST
        )
        self.occgrid.resolution = self.model_args.submap_resolution
        self.occgrid.gridmap = occupancy_grid
        self.occgrid.map_size = np.array([grid_info.height, grid_info.width])

        # Create custom function for requesting submap
        def get_submap(position, orientation):
            """
            position: (2,) numpy matrix
                The offset of the position in meters from the origin
                Index 0 is x, index 1 is y
            orientation: (2,) numpy matrix
                The orientation of the position (does not have to be normalised)
                Index 0 is x, index 1 is y
            """
            # Translate to pixel coordinates
            origin_offset = position.copy().astype(float)
            origin_offset /= self.model_args.submap_resolution
            origin_offset[0] = origin_offset[0] + (78 / self.occgrid.resolution) 
            origin_offset[1] = self.occgrid.gridmap.shape[0] - origin_offset[1] - (40 / self.occgrid.resolution) 
            origin_offset = origin_offset.astype(int)
                    
            # Do bounds-check, returns submaps with zeros if near border or out of bounds
            if (origin_offset[0] < self.model_args.submap_width/ self.model_args.submap_resolution/2 or
                origin_offset[0] > occupancy_grid.shape[1] - self.model_args.submap_width/ self.model_args.submap_resolution/2 or
                origin_offset[1] < self.model_args.submap_height/ self.model_args.submap_resolution/2 or
                origin_offset[1] > occupancy_grid.shape[0] - self.model_args.submap_height/ self.model_args.submap_resolution/2):
                rospy.logerr('Out-of-bounds submap requested!')
                return np.zeros((int(self.model_args.submap_height / self.model_args.submap_resolution), int(self.model_args.submap_width / self.model_args.submap_resolution)))
            
            return occupancy_grid[
                int(origin_offset[1]-self.model_args.submap_height/ self.model_args.submap_resolution/2) : int(origin_offset[1]+self.model_args.submap_height/ self.model_args.submap_resolution/2),
                int(origin_offset[0]-self.model_args.submap_width/ self.model_args.submap_resolution/2) : int(origin_offset[0]+self.model_args.submap_width/ self.model_args.submap_resolution/2)
            ]

        self.get_submap = get_submap

    def callback(self, *roboat_odometry):

        # Check if attribute exists
        if not hasattr(self, '_roboat_pos'):
            self._roboat_pos = np.zeros((2, ), dtype=float)
            self._roboat_vel = np.zeros((2, ), dtype=float)

        curr_pos, curr_vel = {}, {}

        # Read data from msgs and transform velocity to world frame
        for id in range(self.n_agents):
            
            curr_x, curr_y, curr_vx, curr_vy = self.rotate_to_world_frame(roboat_odometry[id])

            curr_pos[id] = np.array([curr_x, curr_y])
            curr_vel[id] = np.array([curr_vx, curr_vy])

        # Update agents' states
        self.agents_pos, self.agents_vel = curr_pos, curr_vel

        # Make prediction
        agents_pos, pred_velocities = self.infer()

        # Publish predictions
        self.publish(agents_pos, pred_velocities)
        
    
    def listener(self):

        # Set up occupancy grid subscriber
        rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self.store_occupancy_grid)

        # Set up odometry subscribers
        roboat_subscribers = []
        for i in range(self.n_agents):
            roboat_subscribers.append( message_filters.Subscriber("/roboat_" + str(i) + "/odometry_enu", Odometry) )
        agent_ids = range(self.n_agents)
        # Set up message filters and callback
        ts = message_filters.ApproximateTimeSynchronizer(roboat_subscribers, 1, 0.1)
        ts.registerCallback(self.callback)

        # Let it spin
        rospy.spin()


    def get_model_args(self, model_name, train_run):
        trained_dir = os.path.join(rospkg.RosPack().get_path(PACKAGE_NAME), 'trained_models')
        model_dir = os.path.join(trained_dir, model_name, train_run)
        convnet_dir = os.path.join(trained_dir, 'autoencoder_with_ped')
        with open(os.path.join(model_dir, 'model_parameters.pkl'), 'rb') as f:
            model_args = pickle.load(f)["args"]
        model_args.model_path = model_dir
        model_args.pretrained_convnet_path = convnet_dir
        model_args.batch_size = MAX_QUERY_AGENTS
        model_args.truncated_backprop_length = 1
        model_args.keep_prob = 1.0
        return model_args


    def load_model(self, model_class, model_args):
        model = model_class(model_args)
        tf_session = tf.Session()
        model.warmstart_model(model_args, tf_session)
        try: model.warmstart_convnet(model_args, tf_session)
        except: rospy.logwarn('Could not warm-start ConvNet')
        rospy.loginfo('Model loaded!')
        return model, tf_session

    def shutdown_callback(self):
      rospy.loginfo('{} was terminated.'.format(self.visual_node_name))


if __name__ == '__main__':

    # Get CLI arguments
    args = rospy.myargv(sys.argv)

    # Start main logic
    node = SVRNN_Predictor('socialvrnn_node', 'Social-VRNN roboat node')

    print("Number of agents: ", node.n_agents)

    # Infer predictions every 'dt' seconds
    rospy.loginfo('Inferring predictions every {} seconds'.format(node.model_args.dt))

    node.listener()


