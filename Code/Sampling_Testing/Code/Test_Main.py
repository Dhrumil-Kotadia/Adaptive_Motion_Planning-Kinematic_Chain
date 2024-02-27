import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

from planner_New import Planner
from robot_New import KinematicChain
from PRM_New import PRM
import time
from RRT_New import log


def load_map(file_path, resolution_scale):
    """Load map from an image and return a 2D binary numpy array
    where 0 represents obstacles and 1 represents free space
    """
    # Load the image with grayscale
    img = Image.open(file_path).convert("L")
    # Rescale the image
    size_x, size_y = img.size
    new_x, new_y = int(size_x * resolution_scale), int(
        size_y * resolution_scale
    )
    img = img.resize((new_x, new_y), Image.ANTIALIAS)

    map_array = np.asarray(img, dtype="uint8")

    # Get bianry image
    threshold = 127
    map_array = 1 * (map_array > threshold)

    # Result 2D numpy array
    return map_array

if __name__ == "__main__":
    # Load the map
    map_2d = load_map("/home/dhrumil/WPI/Sem1/Motion_Planning/Final_Project/Maps/Test_Maps/TM4.png",1)

    # Define the robot with start and goal configurations
    robot = KinematicChain(link_lengths=[150, 150, 150, 150, 150], base=[50, 50])
    
    start = (1.3, -2.5, 1.8, -1.5, 0.8)
    goal = (1.57, -2, 0.5, 0.6, 0.5)

    
    methods = ["uniform", "gaussian", "gaussian", "uniform", "uniform"]

    # For the KinematicChain:

    # method = PRM(sampling_method="uniform", n_configs=100, kdtree_d=np.pi)
    # method = PRM(sampling_method="gaussian", n_configs=100, kdtree_d=np.pi)
    method = PRM(methods, n_configs=100, kdtree_d=np.pi)
    # method = PRM(sampling_method="bridge", n_configs=100, kdtree_d=np.pi)
    # method = PRM(sampling_method="custom_chain", n_configs=100, kdtree_d=np.pi)
    # Initialize the planner
    planner = Planner(method, map_2d, robot)
    solution, pathlength, Number_of_Nodes = planner.plan(start, goal)
    planner.visualize(make_video=False)