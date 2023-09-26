# -*- coding: utf-8 -*-

from math import floor, exp, sqrt
import numpy as np
from src.simulators import social_forces_utils as utils
import sys
from os.path import dirname, abspath
import pylab as pl
import random
import csv
import time

def load_map(data_path):
    try:
        file_path = data_path + '/map.npy'
        if sys.version_info[0] < 3:
            grid_map = np.load(file_path,allow_pickle=True)[()]
        else:
            grid_map = np.load(file_path,encoding='latin1',allow_pickle=True)[()]
        print("Map loaded")
        return grid_map
    except IOError:
        print("ERROR: Cannot load map.")


class ForceModel(object):
    """
    This class is used to simulate and predict social force model.
    """

    def __init__(self, prediction_steps, occ_grid,ped_num,dt=0.1):
        self.desired_speed = 1.0
        self.rel_time = 0.5
        self.delta_t = 0.5
        self.tol = 0.3
        self.ped_num = ped_num
        self.prediction_steps = prediction_steps
        self.occ_grid = occ_grid
        self.dt = dt  # Simulation step in seconds
        try:
            self.map_center = self.occ_grid['Size']/2.0
        except:
            self.map_center = self.occ_grid.map_size
        self.corridor = True

    def goal_force(self, pose, current_speed, goal):
        """
        This function calculates the attractive force towards the goal
        """
        direction = utils.calc_direction(current_pose=pose, desired_pose=goal)
        speed_mod = utils.calc_mod(current_speed)

        force = np.zeros((2))
        force = (direction*self.desired_speed - current_speed)/self.rel_time
        return force

    def repulsive_obj_forces(self, pose, speed, goal):
        """
        This function uses the occupancy grid to calculate the repulsive forces
        of obstacles in a given pose.
        It is not done only once and stored so we can avoid the quantization error
        given by the discretization of the map.
        """
        A = 4.3
        B = 1.07
        l = 0.7

        # Rescale position
        position = (np.array(pose) + self.map_center)/self.occ_grid['Resolution']

        # Get indexes for map
        position_idx = np.ndarray.astype(position, 'int')

        closest_obj = np.array((self.occ_grid['Closest X'][tuple(position_idx)],
                                self.occ_grid['Closest Y'][tuple(position_idx)]))

        rel_distance = pose - closest_obj
        step = - np.array(speed) * self.delta_t
        r_s = rel_distance - step

        mod_rd = utils.calc_mod(rel_distance)
        mod_step = utils.calc_mod(step)
        mod_rs = utils.calc_mod(r_s)

        b = sqrt( pow(mod_rd + mod_rs, 2) - pow(mod_step, 2) )/2

        force = np.zeros((2))
        # if mod_rd != 0 and mod_rs != 0 and mod_step != 0 and b != 0:
        g = A * exp(-b/B) * (mod_rd + mod_rs)/(2*b)
        g = g*(rel_distance/mod_rd + r_s/mod_rs)/2

        force = g


        return force

    def repulsive_ped_force(self, pose, speed, goal, other_ped):
        """
        This function calculates the repulsive forces exerted by other pedestrian.
        """
        A = 4.3
        B = 1.07
        l = 0.7
        # other_ped is in the format [[pose_x, pose_y], [vel_x, vel_y], [goal_x, goal_y]]
        r_ij = np.array(pose) - np.array(other_ped[0])
        s_ij = (np.array(other_ped[1]) - np.array(speed)) * self.delta_t
        r_s_ij = r_ij - s_ij

        mod_r_ij = utils.calc_mod(r_ij)
        mod_r_s_ij = utils.calc_mod(r_s_ij)
        mod_s_ij = utils.calc_mod(s_ij)

        b = sqrt(pow(mod_r_ij + mod_r_s_ij, 2) - pow(mod_s_ij, 2))/2
        force = np.zeros((2))
        if (mod_r_ij != 0) and (mod_r_s_ij != 0) and (mod_s_ij != 0) and (b != 0):
            g = A * exp(-b/B) * (mod_r_ij + mod_r_s_ij) / (2*b)
            g = g * (r_ij/mod_r_ij + r_s_ij/mod_r_s_ij)/2

            e_t_i = utils.calc_direction(current_pose=pose,
                                         desired_pose=goal)

            cos_phi = (e_t_i[0]*r_ij[0]/mod_r_ij) + (e_t_i[1]*r_ij[1]/mod_r_ij)
            w = l + (1 - l)*(1 + cos_phi)/2

            force = g * w

        return force

    def predict(self, ped_id, initial_pose, current_speed, goal, dt):
        """
        This function predicts the next positions and speed of the pedestrian
        using the social force model.
        """
        # data must be in the shape [current + predicted_time_step, ped_index, pose_speed, XY_dims]
        current_pose = initial_pose
        pred_velocities = np.zeros((self.prediction_steps, 2))
        pred_poses = np.zeros((self.prediction_steps, 2))

        for i in range(self.prediction_steps):
            force = self.goal_force(current_pose, current_speed, goal)
            force = force + self.repulsive_obj_forces(pose=current_pose,
                                                      speed=current_speed,
                                                      goal=goal)

            for pp in range(self.ped_num):
                if pp != ped_id:
                    ped_data = [self.current_poses[pp], self.current_speed[pp], self.goals[pp]]
                    force = force + self.repulsive_ped_force(pose=initial_pose,
                                                                         speed=current_speed,
                                                                         goal=goal,
                                                                         other_ped=ped_data)

            current_speed = current_speed + dt * force # Integrate speed
            mod_speed = utils.calc_mod(current_speed)
            max_speed = 1.3 * self.desired_speed
            if mod_speed >= max_speed:
                current_speed = current_speed * max_speed/mod_speed
            current_pose = current_pose + dt * current_speed # Integrate pose

            pred_velocities[i] = current_speed
            pred_poses[i] = current_pose
        return pred_poses, pred_velocities

    def set_initial_pose(self,pose,speed,goals):
        self.initial_pose = pose
        self.current_speed = speed
        self.goals = goals
        self.current_poses = pose

    def get_initial_poses(self, ped_num):
        """
        This functions calculates the random initial positions for the
        pedestrians, making sure they do not spawn into obstacles
        """
        index = 0
        poses = np.zeros((ped_num, 2))
        while index < ped_num:
            if (index % 2):
                pose = np.array((random.uniform(-9.9, -9),
                                 random.uniform(-0.5, 0.5)))
            else:
                pose = np.array((random.uniform(9, 9.9),
                                 random.uniform(-0.5, 0.5)))

            position = (pose + self.map_center)/self.occ_grid['Resolution']
            obstacle = self.occ_grid['Map'][int(position[0]), int(position[1])]
            occupied = utils.check_if_occupied(pose, poses[:index+1])
            if obstacle != 1 and not occupied:
                poses[index] = pose
                index = index + 1
        return poses

    def get_goal(self, pose=None, corridor=False):
        done = False

        if corridor:
            if pose[0] >= 0:
                x_range = np.array([-9.9, -9])
                y_range = np.array([-0.5, 0.5])
            elif pose[0] <= 0:
                x_range = np.array([9, 9.9])
                y_range = np.array([-0.5, 0.5])
        else:
            x_range = np.array([-self.occ_grid['Size'][0]/4,
                                self.occ_grid['Size'][0]/4])
            y_range = np.array([-self.occ_grid['Size'][1]/4,
                                self.occ_grid['Size'][1]/4])

        while not done:
            goal = np.array((random.uniform(x_range[0], x_range[1]),
                             random.uniform(y_range[0], y_range[1])))

            # Check that the goal is not into any obstacle
            position = (goal + self.map_center)/self.occ_grid['Resolution']
            obstacle = self.occ_grid['Map'][int(position[0]), int(position[1])]
            position_idx = np.ndarray.astype(position, 'int')
            closest_obj = np.array((self.occ_grid['Closest X'][tuple(position_idx)],
                                    self.occ_grid['Closest Y'][tuple(position_idx)]))
            if obstacle != 1 or utils.calc_mod(closest_obj - goal) >= 0.5:
                done = True
        return goal

    def simulate(self, ped_num, duration, data_path):
        """
        This function is used to simulate SF model in the given map
        for the given time.
        Input:
            ped_num: Number of pedestrians to simulate
            time: [hours, minutes, seconds]
        """
        self.ped_num = ped_num
        self.min_time = duration[0] * 60 + duration[1]
        sec_time = self.min_time * 60 + duration[2]
        #print("Simulation length: {}h {}m {}s".format(duration[0],
        #                                              duration[1],
        #                                              duration[2]))
        dt = 0.1 # Simulation step in seconds
        sim_elapsed_mins = 0

        self.csv_files = []
        self.writers = []
        self.fieldnames = ['Pedestrian', 'Time s', 'Time ns',
                           'Position X', 'Position Y',
                           'Velocity X', 'Velocity Y',
                           'Goal X', 'Goal Y']

        random.seed()
        counts = np.zeros((self.ped_num))
        poses = self.get_initial_poses(self.ped_num)
        speed = np.zeros((self.ped_num, 2))
        goals = np.zeros((self.ped_num, 2))
        forces = np.zeros((self.ped_num, 2))

        for i in range(self.ped_num):
            goals[i] = self.get_goal(poses[i], corridor=self.corridor)
            # Create logger files
            try:
                self.csv_files.append(open('{}/log_ped{}.csv'.format(data_path, i), 'wb'))
            except:
                print("Cannot open log file number {}. Quitting.".format(i))
                return
            self.writers.append(csv.DictWriter(self.csv_files[i], fieldnames=self.fieldnames))
            self.writers[i].writeheader()
        # Create total log file
        try:
            self.csv_files.append(open('{}/total_log.csv'.format(data_path), 'wb'))
        except:
            print("Cannot open total log file. Quitting.".format(i))
            return
        self.total_log_writer = csv.DictWriter(self.csv_files[-1], fieldnames=self.fieldnames)
        self.total_log_writer.writeheader()

        # Now we are ready to start simulate
        t_start = time.time()
        time_steps = int(sec_time/dt)
        for ts in range(time_steps):
            # NB this could be made for all the peds at once but the force functions
            # for now are designed to work with just one ped at a time
            for ped in range(self.ped_num):
                forces[ped] = self.goal_force(pose=poses[ped],
                                              current_speed=speed[ped],
                                              goal=goals[ped])

                forces[ped] = forces[ped] + self.repulsive_obj_forces(pose=poses[ped],
                                                                      speed=speed[ped],
                                                                      goal=goals[ped])
                for pp in range(self.ped_num):
                    if pp != ped:
                        ped_data = [poses[pp], speed[pp], goals[pp]]
                        forces[ped] = forces[ped] + self.repulsive_ped_force(pose=poses[ped],
                                                                             speed=speed[ped],
                                                                             goal=goals[ped],
                                                                             other_ped=ped_data)
                # Add some noise
                forces[ped] = forces[ped]
            # Once we have all the forces, we update velocity for each ped

            speed = speed + dt * (forces + np.random.normal(0, 0.3, (self.ped_num, 2)))

            for ped in range(self.ped_num):

                mod_speed = utils.calc_mod(speed[ped])
                max_speed = 1.3 * self.desired_speed
                if mod_speed >= max_speed:
                    speed[ped] = speed[ped] * max_speed/mod_speed
                new_pose = poses[ped] + dt * speed[ped]

                # Check if we end up into an obstacle
                # Rescale position
                position = (np.array(new_pose) + self.map_center)/self.occ_grid['Resolution']
                # Get indexes for map
                position = np.ndarray.astype(position, 'int')
                # If we would end up into an obst we do not move
                if self.occ_grid['Map'][tuple(position)] == 1:
                    print("Inside obstacle")
                    new_pose = poses[ped]
                    speed[ped] = np.zeros((2))

                # Update poses
                poses[ped] = new_pose
                # Save data
                data_dict = {}
                data_dict['Pedestrian'] = ped
                data_dict['Time s'] = int(ts*dt)
                data_dict['Time ns'] = int((ts*dt * 10**9)%10**9)
                data_dict['Position X'] = new_pose[0]
                data_dict['Position Y'] = new_pose[1]
                data_dict['Velocity X'] = speed[ped][0]
                data_dict['Velocity Y'] = speed[ped][1]
                data_dict['Goal X'] = goals[ped][0]
                data_dict['Goal Y'] = goals[ped][1]
                self.writers[ped].writerow(data_dict)
                self.total_log_writer.writerow(data_dict)
                # Update goal
                if utils.calc_mod(goals[ped] - new_pose) <= self.tol:
                    goals[ped] = self.get_goal(new_pose, corridor=self.corridor)
                    counts[ped] = 0
                # Check if stuck
                if utils.calc_mod(speed[ped]) <= 0.01:
                    counts[ped] = counts[ped] + 1
                    if counts[ped] == 5:
                        goals[ped] = self.get_goal(new_pose, corridor=self.corridor)
                        counts[ped] = 0
                else: # This way we just count when they are consecutive
                    counts[ped] = 0

            # Log passing time
            rt_time = time.time() - t_start
            if (ts*dt/60 - sim_elapsed_mins) >= 1:
                sim_elapsed_mins = int(ts*dt/60)
                fact = ts*dt / rt_time
                rt_time = time.gmtime(rt_time)
                print("Elapsed simulation time since start: {} mins".format(sim_elapsed_mins))
                print("Elapsed real time since start: {}h {}m {}s".format(rt_time.tm_hour,
                                                                          rt_time.tm_min,
                                                                          rt_time.tm_sec))
                print("Sim to RT factor: {}".format(fact))
                print("")

        #
        for i in range(self.ped_num + 1):
            self.csv_files[i].close()

        print("Simulation finished.")
        rt_time = time.gmtime(time.time() - t_start)
        print("Total real time: {}h {}m {}s".format(rt_time.tm_hour,
                                                    rt_time.tm_min,
                                                    rt_time.tm_sec))
    def simulate_ped(self, ped_id,ini_pose,goal,duration,other_agents_pos):
        """
        This function is used to simulate SF model in the given map
        for the given time.
        Input:
            ped_num: Number of pedestrians to simulate
            time: [hours, minutes, seconds]
        """

        self.min_time = duration[0] * 60 + duration[1]
        sec_time = self.min_time * 60 + duration[2]
        print("Simulation length: {}h {}m {}s".format(duration[0],
                                                      duration[1],
                                                      duration[2]))
        dt = 0.3 # Simulation step in seconds
        sim_elapsed_mins = 0

        random.seed()
        counts = np.zeros((self.ped_num))
        poses = np.expand_dims(ini_pose[0:2].copy(),0)
        speed = np.expand_dims(ini_pose[2:].copy(),0)
        goals = np.zeros((self.ped_num, 2))
        forces = np.zeros((self.ped_num, 2))

        for i in range(self.ped_num):
            goals[i] = goal

        # Now we are ready to start simulate
        t_start = time.time()
        time_steps = int(sec_time/dt)
        traj = np.zeros((time_steps,2))
        velocities = np.zeros((time_steps,2))
        velocities[0,:] = speed
        traj[0,:] = ini_pose[0:2]
        for ts in range(time_steps-1):
            # NB this could be made for all the peds at once but the force functions
            # for now are designed to work with just one ped at a time
            ped=0
            forces[ped] = self.goal_force(pose=poses[ped],
                                         current_speed=speed[ped],
                                         goal=goals[ped])

            forces[ped] = forces[ped] + self.repulsive_obj_forces(pose=poses[ped],
                                                                  speed=speed[ped],
                                                                  goal=goals[ped])
            for pp in range(len(other_agents_pos)):
                agent_pos = np.reshape(other_agents_pos[pp],2)
                ped_data = [agent_pos, np.zeros((2)), agent_pos]
                forces[ped] = forces[ped] + self.repulsive_ped_force(pose=poses[ped],
                                                                     speed=speed[ped],
                                                                     goal=goals[ped],
                                                                     other_ped=ped_data)
            # Add some noise
            forces[ped] = forces[ped]
            # Once we have all the forces, we update velocity for each ped

            speed = speed + dt * (forces + np.random.normal(0, 0.01, (self.ped_num, 2)))

            mod_speed = utils.calc_mod(speed[ped])
            max_speed = 1.3 * self.desired_speed
            if mod_speed >= max_speed:
                speed[ped] = speed[ped] * max_speed/mod_speed
            new_pose = poses[ped] + dt * speed[ped]

            # Check if we end up into an obstacle
            # Rescale position
            position = (np.array(new_pose) + self.map_center)/self.occ_grid['Resolution']
            # Get indexes for map
            position = np.ndarray.astype(position, 'int')
            # If we would end up into an obst we do not move
            if self.occ_grid['Map'][tuple(position)] == 1:
                new_pose = poses[ped]
                speed[ped] = np.zeros((2))

            # Update poses
            poses[ped] = new_pose

            # Update goal
            if utils.calc_mod(goals[ped] - new_pose) <= self.tol:
                goals[ped] = self.get_goal(new_pose, corridor=self.corridor)
                counts[ped] = 0
            # Check if stuck
            if utils.calc_mod(speed[ped]) <= 0.01:
                counts[ped] = counts[ped] + 1
                if counts[ped] == 5:
                    goals[ped] = self.get_goal(new_pose, corridor=self.corridor)
                    counts[ped] = 0
            else: # This way we just count when they are consecutive
                counts[ped] = 0

            traj[ts+1,:]=poses[ped]
            velocities[ts+1,:]=speed[ped]
        return traj, velocities
    
    def simulation_step(self,sec_time):
        """
        This function is used to simulate one step of SF model in the given map
        for the given time.
        Input:
            ped_num: Number of pedestrians to simulate
            time: [hours, minutes, seconds]
        """

        forces = np.zeros((self.ped_num, 2))

        # Now we are ready to start simulate
        time_steps = int(sec_time/self.dt)
        for ts in range(time_steps):
            # NB this could be made for all the peds at once but the force functions
            # for now are designed to work with just one ped at a time
            for ped in range(self.ped_num):
                forces[ped] = self.goal_force(pose=self.current_poses[ped],
                                              current_speed=self.current_speed[ped],
                                              goal=self.goals[ped])

                forces[ped] = forces[ped] + self.repulsive_obj_forces(pose=self.current_poses[ped],
                                                                      speed=self.current_speed[ped],
                                                                      goal=self.goals[ped])
                for pp in range(self.ped_num):
                    if pp != ped:
                        ped_data = [self.current_poses[pp], self.current_speed[pp], self.goals[pp]]
                        forces[ped] = forces[ped] + self.repulsive_ped_force(pose=self.current_poses[ped],
                                                                             speed=self.current_speed[ped],
                                                                             goal=self.goals[ped],
                                                                             other_ped=ped_data)
                # Add some noise
                forces[ped] = forces[ped]
            # Once we have all the forces, we update velocity for each ped

            self.current_speed = self.current_speed + self.dt * (forces + np.random.normal(0, 0.1, (self.ped_num, 2)))

            for ped in range(self.ped_num):

                mod_speed = utils.calc_mod(self.current_speed[ped])
                max_speed = 1.3 * self.desired_speed
                if mod_speed >= max_speed:
                    self.current_speed[ped] = self.current_speed[ped] * max_speed/mod_speed
                new_pose = self.current_poses[ped] + self.dt * self.current_speed[ped]

                # Check if we end up into an obstacle
                # Rescale position
                position = (np.array(new_pose) + self.map_center)/self.occ_grid['Resolution']
                # Get indexes for map
                position = np.ndarray.astype(position, 'int')
                # If we would end up into an obst we do not move
                if self.occ_grid['Map'][tuple(position)] == 1:
                    print("Inside obstacle")
                    new_pose = self.current_poses[ped]
                    self.current_speed[ped] = np.zeros((2))

                # Update poses
                self.current_poses[ped] = new_pose

                # Update goal
                if utils.calc_mod(self.goals[ped] - new_pose) <= self.tol:
                    print("Goal reached")
                # Check if stuck
                if utils.calc_mod(self.current_speed[ped]) <= 0.01:
                    print("Stuck")

        return self.current_poses , self.current_speed

# All the stuff here is for test
if __name__ == "__main__":

    data_path ="../../data/multipath"

    grid = load_map(data_path)
    model = ForceModel(predicted=10, occ_grid=grid,ped_num=2)

    model.simulate(2, [5,0,0], data_path)