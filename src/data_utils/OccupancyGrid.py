import numpy as np
import cv2 as cv
import src.data_utils.Support as sup
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class OccupancyGrid():
	"""
	Occupancy grid class for capturing static object information.
	This occupancy grid is aligned with the Cartesian coordinate frame: 
		index 0: x-axis
		index 1: y-axis
	"""

	def __init__(self):
		self.gridmap = None
		self.resolution = None
		self.map_size = None
		#self.center = np.array([0.0, 0.0])
		self.center = np.array([-78.0, -40.0])
		
	def getIdx(self, pos_x, pos_y):
		"""
		Get indices of position. 
		pos_x and pos_y are the positions w.r.t. the center of the map.
		"""
		#idx_x = int((pos_x + float(self.map_size[0]) / 2.0) / self.resolution)
		#idx_y = int((pos_y + float(self.map_size[1]) / 2.0) / self.resolution)
		# modified
		#idx_x = int(pos_x  / self.resolution)
		#idx_y = int(pos_y / self.resolution)

		# Custom function to extract correct indices from Herengracht map
		idx_y = int((pos_x / self.resolution) + (self.center[0]*-1 / self.resolution)) 
		idx_x = self.map_size[1] - int((pos_y / self.resolution) + (self.center[1]*-1 / self.resolution))
		#self.map_size[1] - 
		#print("Calculated indices: ", idx_x, idx_y)
		
		# Projecting index on map if out of bounds
		idx_x = max(0, min(idx_x, self.map_size[0] / self.resolution))
		idx_y = max(0, min(idx_y, self.map_size[1] / self.resolution))
	 
		"""

		print("Actual indices: ", idx_x, idx_y)
		submap = []
		submap = self.getFrontSubmapByIndices(idx_y, idx_x, 60,60)[0]

		fig, (ax1, ax2) = plt.subplots(1,2)
		
		print("Submap shape 1: ", submap.shape)
		sup.plot_grid_roboat(ax1, self.center, self.gridmap, self.resolution,
									self.map_size)
		ax1.plot(pos_x, pos_y, marker="o", markersize=5) 
		#sup.plot_grid(ax2, np.array([0.0]), submap, self.resolution, np.array([60,60]))
		
		#rect = patches.Rectangle((pos_x,pos_y),4.86,4.86, edgecolor='r', facecolor="none")
		#ax1.add_patch(rect)
		
		print("Submap shape 2: ", submap.shape)
		
		ax2.imshow(submap, cmap='gray_r')
		#ax2.axis("on")
		#ax2.set_xlabel('x [m]',fontsize=26)
		#ax2.set_ylabel('y [m]',fontsize=26)
		#ax2.set_xlim([-6.0, 6.0])
		#ax2.set_ylim([-6.0, 6.0])
		#ax2.set_aspect('equal')
	
		plt.show()

		print("Submap shape 3: ", submap.shape)
		
		"""

		return idx_x, idx_y
		
	
	def getSubmapByIndices(self, center_idx_x, center_idx_y, span_x, span_y):
		"""
		Extract a submap of span (span_x, span_y) around 
		center index (center_idx_x, center_idx_y)
		"""
		debug_info = {}
		start_idx_x = max(0, int(center_idx_x - np.floor(span_x / 2)))
		start_idx_y = max(0, int(center_idx_y - np.floor(span_y / 2)))
		
		# Compute end indices (assure size of submap is correct, if out pf bounds)
		max_idx_x = self.gridmap.shape[0] - 1
		max_idx_y = self.gridmap.shape[1] - 1
		
		end_idx_x = start_idx_x + span_x
		if end_idx_x > max_idx_x:
			end_idx_x = max_idx_x
			start_idx_x = end_idx_x - span_x
		end_idx_y = start_idx_y + span_y
		if end_idx_y > max_idx_y:
			end_idx_y = max_idx_y
			start_idx_y = end_idx_y - span_y
		
		# Collect debug information
		debug_info["start_x"] = start_idx_x
		debug_info["start_y"] = start_idx_y
		debug_info["end_x"] = end_idx_x
		debug_info["end_y"] = end_idx_y

		#print(self.gridmap.shape[0], self.gridmap.shape[1])
		#print("")
		#print(start_idx_x, end_idx_x)
		#print(start_idx_y, end_idx_y)

		#print("Extracted submap (startx, endx, starty, endy): ",start_idx_x, end_idx_x, start_idx_y, end_idx_y)

		#self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y] = 0.5

		#print("Shape in func: ", self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y].shape)

		#self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y]

		submap = self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y]

		submap_rotated = list(zip(*submap[::-1]))
		
		return submap_rotated, debug_info
		
	def getSubmapByCoords(self, center_pos_x, center_pos_y, size_x, size_y):
		"""
		Get submap around specified coordinates. 
		The sizes in x and y direction are within the same coordinate frame as the center coordinates.
		"""
		center_idx_x, center_idx_y = self.getIdx(center_pos_x, center_pos_y)
		span_x = int(np.ceil(size_x / self.resolution))
		span_y = int(np.ceil(size_y / self.resolution))

		#print(center_pos_x, center_pos_y, size_x, size_y)
		#print(center_idx_x, center_idx_y, span_x, span_y)
		
		return self.getSubmapByIndices(center_idx_x, center_idx_y, span_x, span_y)[0]

	def getFrontSubmap(self, center, velocity, size_x, size_y):
		"""
		Get submap around specified coordinates.
		The sizes in x and y direction are within the same coordinate frame as the center coordinates.
		"""
		center_idx_x, center_idx_y = self.getIdx(center[0], center[1])
		span_x = int(np.ceil(size_x / self.resolution))
		span_y = int(np.ceil(size_y / self.resolution))

		if velocity[0] > 0.1 :
			center_idx_x += span_x
		elif velocity[0] < -0.1:
			center_idx_x -= span_x

		return self.getSubmapByIndices(center_idx_x, center_idx_y, span_x, span_y)[0]

	def getFrontSubmapByIndices(self, center_idx_x, center_idx_y, span_x, span_y):
		"""
		Extract a submap of span (span_x, span_y) around
		center index (center_idx_x, center_idx_y)
		"""
		debug_info = {}
		start_idx_x = max(0, int(center_idx_x ))
		start_idx_y = max(0, int(center_idx_y ))

		# Compute end indices (assure size of submap is correct, if out pf bounds)
		max_idx_x = self.gridmap.shape[0] - 1
		max_idx_y = self.gridmap.shape[1] - 1

		end_idx_x = start_idx_x + 2*span_x
		if end_idx_x > max_idx_x:
			end_idx_x = max_idx_x
			start_idx_x = end_idx_x - span_x
		end_idx_y = start_idx_y + span_y
		if end_idx_y > max_idx_y:
			end_idx_y = max_idx_y
			start_idx_y = end_idx_y - span_y

		return self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y], debug_info

	def getFrontSubmapByCoords(self, center_pos_x, center_pos_y, size_x, size_y,grid_map):
		"""
		Get submap around specified coordinates.
		The sizes in x and y direction are within the same coordinate frame as the center coordinates.
		"""
		center_idx_x, center_idx_y = self.getIdx(center_pos_x, center_pos_y)
		span_x = int(np.ceil(size_x / self.resolution))
		span_y = int(np.ceil(size_y / self.resolution))
		grid = np.zeros((span_x,span_y))

		start_idx_x = max(0, int(center_idx_x))
		start_idx_y = max(0, int(center_idx_y-span_y/2))

		# Compute end indices (assure size of submap is correct, if out pf bounds)
		max_idx_x = self.gridmap.shape[0] - 1
		max_idx_y = self.gridmap.shape[1] - 1

		end_idx_x = start_idx_x + span_x
		end_idx_y = start_idx_y + span_y
		if end_idx_x > max_idx_x:
			end_idx_x = max_idx_x
		if end_idx_y > max_idx_y:
			end_idx_y = max_idx_y
		dx = end_idx_x-start_idx_x
		dy = min(0,span_y/2-start_idx_y)
		if start_idx_y+span_y<max_idx_y:
			dy_end = span_y
		else:
			dy_end = max_idx_y-start_idx_y

		grid[0:dx,dy:dy_end] = grid_map[start_idx_x:end_idx_x,start_idx_y:end_idx_y]
		return grid
		