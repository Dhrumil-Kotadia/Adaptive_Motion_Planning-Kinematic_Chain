
from token import NUMBER
import numpy as np
from PIL import Image

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

    # For the KinematicChain
    # Uncomment the method that is needed and comment the rest
    
    # method = PRM(sampling_method="uniform", n_configs=100, kdtree_d=np.pi)
    method = PRM(sampling_method="gaussian", n_configs=100, kdtree_d=np.pi)
    # method = PRM(sampling_method="bridge", n_configs=100, kdtree_d=np.pi)

    Base_Locations = [[20, 20], [150, 20], [280,20], [280, 150], [280, 280], [150,280], [20,280], [20,150]]
    Start_Locations = [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1.57, 0, 0, 0, 0), (1.57, 0, 0, 0, 0), (4.71, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1.57, 0, 0, 0, 0)]
    Goal_Locations = [(1.57, 0, 0, 0, 0), (3.14, 0, 0, 0, 0), (3.14, 0, 0, 0, 0), (4.71, 0, 0, 0, 0), (3.14, 0, 0, 0, 0), (-3.14, 0, 0, 0, 0), (-1.57, 0, 0, 0, 0), (-1.57, 0, 0, 0, 0)]

    for map_counter in range(23):                                                                                               # Change the number based on the number of maps
        map_counter += 19
        temp_path_string = "/home/dhrumil/WPI/Sem1/Motion_Planning/Final_Project/Maps/Obstacles/Map"                            # Change the path according to system
        path_string = temp_path_string+str(map_counter)+".png"
        log.I("#########################")
        log.I("Map: "+str(map_counter))
        log.I("#########################")
        map_2d = load_map(path_string,1)                                                                                        # Load the map file as mentioned in the string. 1 indicates scaling
        file = open("/home/dhrumil/WPI/Sem1/Motion_Planning/Final_Project/Gaussian.txt", "a")                                   # Open text file to append data
        file.write('Map: '+str(map_counter))
        file.write('\n')
        file.write('File: '+path_string)
        file.write('\n')
        file.write('\n')
        for location_counter in range(8):                                                                                       # Location counter decides the Base, Start and Goal based on the respective arrays
            robot = KinematicChain(link_lengths=[25, 25, 25, 25, 25], base=Base_Locations[location_counter])                    # Chain with 5 Links and base decided by location counter
            file.write('\n')
            file.write("-------------------------")
            file.write('\n')
            file.write('Base: '+str(Base_Locations[location_counter]))
            file.write('\n')
            file.write('Start: '+str(Start_Locations[location_counter])+'; Goal: '+str(Goal_Locations[location_counter]))
            file.write('\n')
            file.write('\n')
            file.write("-------------------------")
            file.write('\n')
            start = Start_Locations[location_counter]
            goal = Goal_Locations[location_counter]
            for counter in range(1):                                                                                           # Number of iterations for each position. Change if necessary
                tic = time.perf_counter()                                                                                       # Start Timer to measure Time Taken
                planner = Planner(method, map_2d, robot)
                # planning
                solution, pathlength, Number_of_Nodes = planner.plan(start, goal)
                toc = time.perf_counter()                                                                                       # Stop Timer
                time_taken = int(toc-tic)                                                                                       # Total Time Taken for this execution
                # planner.visualize(make_video=False)
                log.I("-------------------------")
                log.I("Map: "+str(map_counter))
                log.I("Base Position: "+str(location_counter+1))
                log.I('Time Taken: '+str(time_taken))
                log.I('Number of Nodes: '+str(Number_of_Nodes))
                log.I("-------------------------")
                file.write('Time Taken: '+str(time_taken)+'; Number of Nodes: '+str(Number_of_Nodes)+'; Path Length: '+str(pathlength))
                file.write('\n')
            file.write("-------------------------")
        file.write('\n')
        file.close
