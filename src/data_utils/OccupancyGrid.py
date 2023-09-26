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
		self.center = np.array([None, None])
		
	def getIdx(self, pos_x, pos_y):
		"""
		Get indices of position. 
		pos_x and pos_y are the positions w.r.t. the center of the map.
		"""

		# Custom function to extract correct indices from Herengracht map
		idx_y = int((pos_x / self.resolution) + (self.center[0]*-1 / self.resolution)) 
		idx_x = self.map_size[1] - int((pos_y / self.resolution) + (self.center[1]*-1 / self.resolution))

		
		# Projecting index on map if out of bounds
		idx_x = max(0, min(idx_x, self.map_size[0] / self.resolution))
		idx_y = max(0, min(idx_y, self.map_size[1] / self.resolution))

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

		#print("Shape in func: ", self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y].shape)

		submap = self.gridmap[start_idx_x:end_idx_x, start_idx_y:end_idx_y]
		
		"""
		print(self.resolution)
		print(center_idx_x, center_idx_y)
		print(span_x, span_y)

		plt.figure()
		plt.imshow(self.gridmap, cmap="gray_r")
		plt.show()

		plt.figure()
		plt.imshow(submap, cmap="gray_r")
		plt.show()
		"""
		
		return submap, debug_info
		
	def getSubmapByCoords(self, center_pos_x, center_pos_y, size_x, size_y):
		"""
		Get submap around specified coordinates. 
		The sizes in x and y direction are within the same coordinate frame as the center coordinates.
		"""
		center_idx_x, center_idx_y = self.getIdx(center_pos_x, center_pos_y)
		span_x = int(size_x / self.resolution)
		span_y = int(size_y / self.resolution)

		#print("coordinates: ", center_pos_x, center_pos_y)
		
		return self.getSubmapByIndices(center_idx_x, center_idx_y, span_x, span_y)[0]
