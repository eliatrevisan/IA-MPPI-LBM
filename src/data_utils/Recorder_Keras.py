import os
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse
# from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d.axes3d as p3

from src.data_utils.DataHandler_Keras_v2 import prediction_to_trajectory

class Recorder(object): 
    def __init__(self, all_trajectories, all_predictions, args, save = False, display = True, figsize = (8,6), nframes = 10000): 
        self.dt = args.dt
        self.all_predictions = all_predictions
        self.all_trajectories = all_trajectories
        self.args = args
        # self.fps = int(1/self.dt)
        self.fps = int(10/self.dt)
        
        self.fig = plt.figure(figsize=figsize)
        self.ax = plt.axes(xlim=(-15, 15), ylim=(-15, 15))
        
        self.stream = self.data_stream()

        total_frames = 0
        for prediction in all_predictions:
            total_frames += prediction.shape[0]

        self.nframes = min(nframes, total_frames) 

        self.ani = animation.FuncAnimation(self.fig, self.animate, frames = self.nframes, init_func=self.setup_plot, interval=1/self.fps, blit=False, repeat=False)

        if save:
            self.save_animation(self.ani)

        if display:
            plt.show()


    def setup_plot(self):
        # Axis setup
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        pedestrian, = self.ax.plot([], [], marker="o", color='b')
        pedestrian_goal, = self.ax.plot([], [],  marker="D", color='b')
        robot, = self.ax.plot([], [], marker="o", color='r')
        robot_goal, = self.ax.plot([], [], marker="D", color='r')
        complete_path, = self.ax.plot([], [], marker="o", fillstyle='none', markersize=5, lw=1.5)
        predicted_path, = self.ax.plot([], [], marker="x", markersize=5, lw=1.5, color='g')
        self.objects = {
            "pedestrian": pedestrian,
            "pedestrian_goal": pedestrian_goal,
            "robot": robot,
            "robot_goal": robot_goal,
            "complete_path": complete_path,
            "predicted_path": predicted_path
        }
        return self.objects,

    def animate(self, iteration):
        data = next(self.stream)
        
        prediction = prediction_to_trajectory(data["pedestrian_position"], data["prediction"], self.args)
        
        self.objects["pedestrian"].set_data(data["pedestrian_position"][0], data["pedestrian_position"][1])
        self.objects["robot"].set_data(data["robot_position"][0], data["robot_position"][1])
        self.objects["pedestrian_goal"].set_data(data["pedestrian_goal"][0], data["pedestrian_goal"][1])
        self.objects["robot_goal"].set_data(data["robot_goal"][0], data["robot_goal"][1])
        self.objects["complete_path"].set_data(data["complete_path"][:, 0], data["complete_path"][:, 1])
        self.objects["predicted_path"].set_data( prediction[:, 0], prediction[:, 1] )
        
        return self.objects,

    def data_stream(self):
        """ Generator for the data to be plotted """
        while True:
            for prediction_array, trajectory in zip(self.all_predictions, self.all_trajectories):
                complete_path = np.zeros((len(trajectory), 2))
                for idx, datapoint in enumerate(trajectory):
                    complete_path[idx, :] = datapoint['pedestrian_state']['position']
                    
                for prediction, datapoint in zip(prediction_array, trajectory):
                    yield {
                        "prediction": prediction,
                        "pedestrian_position": datapoint['pedestrian_state']['position'],
                        "robot_position": datapoint['robot_state'],
                        "pedestrian_goal": datapoint["pedestrian_goal_position"],
                        # "robot_goal": trajectory[-1]["robot_state"][0:2],
                        "robot_goal": datapoint["goal_position"],
                        "complete_path": complete_path
                    }

    def save_animation(self, ani):
        self.writer =  animation.writers['ffmpeg']
        self.writer = self.writer(fps=self.fps, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        video_dir = os.path.join(self.args.model_path, self.args.scenario)
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        video_name = os.path.join(video_dir, 'video.mp4')
        self.ani.save(video_name, writer=self.writer)
