# -*- coding: utf-8 -*-

from math import ceil
import numpy as np
import os
import cv2 as cv
import csv
import pylab as pl
from PIL import ImageDraw
from PIL import Image
from numpngw import write_png

# TODO Merge this functions into Support.py
# ------------------------------------------------------------------------------
def create_map_from_obstacles(map_shape=[30, 30],
                              map_resolution=0.1,
                              save_data_path='../data'):
  grid_map = {}
  grid_map['Resolution'] = map_resolution
  grid_map['Size'] = np.array(map_shape)
  grid_map['Map'] = np.zeros((int(grid_map['Size'][0] / grid_map['Resolution']),
                              int(grid_map['Size'][1] / grid_map['Resolution'])))
  map_center = grid_map['Size'] /2 # other datasets /2

  xx, yy = np.mgrid[:map_shape[0] / map_resolution, :map_shape[1] / map_resolution]
  # First read and draw cylinders
  with open("../objects/cylinders.csv", "rb") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
      if i == 0:
        continue
      center = np.array((float(line[0]) + map_center[0],
                         float(line[1]) + map_center[1])) / map_resolution
      circle = (xx - int(center[0]))**2 + (yy - int(center[1]))**2
      grid_map['Map'][circle <= int(float(line[2]) / map_resolution)**2] = 1

  # Now go with the rectangles
  with open("../objects/rectangles.csv", "rb") as f:
    reader = csv.reader(f, delimiter=",")
    img = Image.fromarray(grid_map['Map'])
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(reader):
      if i == 0:
        continue
      center = np.array((float(line[0]), float(line[1]))) / map_resolution
      shape = np.array((float(line[2]), float(line[3]))) / map_resolution
      theta = (np.pi / 180.0) * float(line[4])  # In radians
      rect = np.array(((-shape[1] / 2., -shape[0] / 2.),
                       (-shape[1] / 2., shape[0] / 2.),
                       (shape[1] / 2., shape[0] / 2.),
                       (shape[1] / 2., -shape[0] / 2.)))

      R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

      rect = np.dot(rect, R) + center[::-1] + map_center / map_resolution
      draw.polygon([tuple(p) for p in rect], fill=1)
    grid_map['Map'] = np.asarray(img)
  print("Occupancy grid done.")

  # Calculate closest obstacles
  grid_map['Closest X'] = np.zeros_like(grid_map['Map']) + 10000
  grid_map['Closest Y'] = np.zeros_like(grid_map['Map']) + 10000

  # Get idx in map of places where there are obstacles
  obst_idx = np.array(np.where(grid_map['Map'] == 1))

  for xx in range(int(grid_map['Size'][0] / grid_map['Resolution'])):
    for yy in range(int(grid_map['Size'][1] / grid_map['Resolution'])):
      distances = np.sqrt(np.sum(np.square(obst_idx - np.expand_dims([xx, yy], axis=1)), axis=0))
      closest_obj = np.argmin(distances)
      # Get index in map of the closest_obj
      grid_map['Closest X'][xx, yy] = obst_idx[0, closest_obj]
      grid_map['Closest Y'][xx, yy] = obst_idx[1, closest_obj]

  grid_map['Closest X'] = grid_map['Closest X'] * grid_map['Resolution'] - map_center[0]
  grid_map['Closest Y'] = grid_map['Closest Y'] * grid_map['Resolution'] - map_center[1]

  print("Closest obstacles positions calculated.")
  print("Saving map...")
  np.save(os.path.join(save_data_path, 'map'), grid_map)
  print("Map saved.")
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def create_map_from_png(load_data_path='../data/datasets/ewap_dataset/seq_eth',
                        save_data_path='../data'):
  # Create grid for SF model
  print("create_map_from_png")
  grid_map = {}
  grid_map['Resolution'] = 0.1
  grid_map['Size'] = np.array([20., 7.])  # map size in [m]
  grid_map['Map'] = np.zeros((int(grid_map['Size'][0] / grid_map['Resolution']),
                              int(grid_map['Size'][1] / grid_map['Resolution'])))

  map_center = grid_map['Size'] / 2. # hack for my dataset
  H = np.genfromtxt(os.path.join(load_data_path, 'H.txt'),
                    delimiter='  ',
                    unpack=True).transpose()

  # Extract static obstacles
  obst_threshold = 200
  static_obst_img = cv.imread(os.path.join(load_data_path, 'map.png'), 0)
  #static_obst_img = cv.imread(os.path.join(load_data_path, 'map.png'), 0)
  obstacles = np.zeros([0, 3])
  #grid = (static_obst_img/255*-1)+1
  #static_obst_img = np.transpose(static_obst_img)
  for xx in range(static_obst_img.shape[0]):
    for yy in range(static_obst_img.shape[1]):
      if static_obst_img[xx, yy] > obst_threshold:
        obstacles = np.append(obstacles,
                              np.dot(H, np.array([[xx], [yy], [1]])).transpose(),
                              axis=0)
  # Compute obstacles in 2D
  obstacles_2d = np.zeros([obstacles.shape[0], 2])
  obstacles_2d[:, 0] = obstacles[:, 0] / obstacles[:, 2]
  obstacles_2d[:, 1] = obstacles[:, 1] / obstacles[:, 2]

  # Get obstacle idx on map
  obst_idx = []
  for obst_ii in range(obstacles_2d.shape[0]):
    obst_idx.append(idx_from_pos(obstacles_2d[obst_ii, 0],
                    obstacles_2d[obst_ii, 1],
                    map_center,grid_map['Resolution']))
    grid_map['Map'][obst_idx[-1]] = 1

  grid_map['Closest X'] = np.zeros_like(grid_map['Map']) + 10000
  grid_map['Closest Y'] = np.zeros_like(grid_map['Map']) + 10000
  # Calculate closest obstacle
  obst_idx = np.array(obst_idx)
  for xx in range(int(grid_map['Size'][0] / grid_map['Resolution'])):
    for yy in range(int(grid_map['Size'][1] / grid_map['Resolution'])):
      delta_idx = obst_idx - np.array([xx, yy])
      distances = np.sqrt(np.sum(np.square(delta_idx), axis=1))
      closest_obj = np.argmin(distances)
      grid_map['Closest X'][xx, yy] = obst_idx[closest_obj][0]
      grid_map['Closest Y'][xx, yy] = obst_idx[closest_obj][1]

  grid_map['Closest X'] = grid_map['Closest X'] * grid_map['Resolution'] - map_center[0]
  grid_map['Closest Y'] = grid_map['Closest Y'] * grid_map['Resolution'] - map_center[1]

  np.save(os.path.join(save_data_path, 'map'), grid_map)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def rotateGridAroundCenter(grid, angle):
  # Rotate grid into direction of initial heading
  grid = grid.copy()
  rows, cols = grid.shape
  M = cv.getRotationMatrix2D(center=(rows/2, cols/2), angle=angle, scale=1)
  grid = np.array(cv.warpAffine(grid, M, (rows, cols)))

  grid[grid > 0] = 1 # This way we remove artifacts around borders
  grid[grid < 1] = 0 # This way we remove any negative value

  return grid
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def idx_from_pos(x, y, center, res=0.1):
  idx_x = round((x + float(center[0])) / res)
  idx_y = round((y + float(center[1])) / res)

  # Projecting index on map if out of bounds
  idx_x = max(0, min(idx_x, -1 + center[0] * 2. / res))
  idx_y = max(0, min(idx_y, -1 + center[1] * 2. / res))
  return int(idx_x), int(idx_y)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calc_mod(vect):
  vect = np.array(vect)
  return np.sqrt(pow(vect[0], 2) + pow(vect[1], 2))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calc_angle(x, y):
  angles = np.array(np.arctan2(y, x) * 180 / np.pi)
  angles[angles == 180.] = -180. # This way output is in [-180,180) and not (-180,180]
  return angles
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calc_rotation_matrix(angle):
  """
  This function calculates the active rotation matrix.
  Thus if you want to rotate the ref system and not the point you shoud use
  its transpose.
  Thus R = [c, -s; s, c]
  """
  theta = np.array(np.radians(angle))
  c, s = np.cos(theta), np.sin(theta)
  R = np.zeros((len(theta), 2, 2))
  R[:, 0, 0] = c
  R[:, 0, 1] = -s
  R[:, 1, 0] = s
  R[:, 1, 1] = c
  return R
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calc_direction(current_pose, desired_pose):
  """
  This function calculates the normalized vector going from the current pose
  towards the desired pose.
  """
  tmp = np.array(desired_pose) - np.array(current_pose)
  mod_e_t = calc_mod(tmp)

  e_t = np.zeros((2))

  if mod_e_t is not 0:
    e_t = tmp / mod_e_t

  return e_t
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def check_if_occupied(pose, other_poses):
  """
  This function is used in the SF model to check if a position
  is already occupied by another pedestrian.
  """
  ped_radius = 0.2
  for index, ped_pose in enumerate(other_poses):
    distance = calc_mod(pose - ped_pose)
    if distance <= 2 * ped_radius:
      return True

  return False
# ------------------------------------------------------------------------------

def write_map_png(file, grid):
  grid = rotateGridAroundCenter(grid,90)
  #grid = (grid-1)*-255
  img = grid.astype(np.uint8)
  write_png(file, img, bitdepth=1)

# ------------------------------------------------------------------------------
def load_map(data_path):
  try:
    file_path = data_path + '/map.npy'
    grid_map = np.load(file_path)[()]
    print("Map loaded")
    return grid_map
  except IOError:
    print("ERROR: Cannot load map.")
    return False
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def show_map(grid_map):
  i = 1
  for k in grid_map.keys():
    if k in ["Size", "Resolution"]:
      continue
    fig = pl.figure(i)
    pl.imshow(grid_map[k], cmap=pl.cm.ocean)
    pl.colorbar()
    pl.title(k)
    i = i + 1
  pl.show()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calc_direction(current_pose, desired_pose):
    """
    This function calculates the normalized vector going from the current pose
    towards the desired pose.
    """
    tmp = np.array(desired_pose) - np.array(current_pose)
    mod_e_t = calc_mod(tmp)

    e_t = np.zeros((2))

    if mod_e_t != 0:
        e_t = tmp/mod_e_t

    return e_t
# ------------------------------------------------------------------------------
if __name__ == "__main__":
  create_map_from_png(
    load_data_path='/media/bdebrito/7697ec91-468a-4763-b1c3-135caa7f5aed/home/code/lstm_pedestrian_prediction/data/multipath/corridor_3peds',
  save_data_path = '/media/bdebrito/7697ec91-468a-4763-b1c3-135caa7f5aed/home/code/lstm_pedestrian_prediction/data/multipath/corridor_3peds')
  grid_map = load_map("/media/bdebrito/7697ec91-468a-4763-b1c3-135caa7f5aed/home/code/lstm_pedestrian_prediction/data/multipath/corridor_3peds")
  show_map(grid_map)
  print("DOne!")