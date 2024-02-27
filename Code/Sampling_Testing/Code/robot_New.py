import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as manimation
from RRT_New import log
from math import sqrt

from utils import endpoints_to_edges, angle_diff, interpolate_angle
from utils import is_in_polygon, is_intersecting, line_intersection


class Robot:
    """A parent class for all robots"""

    def __init__(self, limits):
        """Initialize by providing limits of each degree of freedom"""
        # Limits in each dof, each limit is defined as (lower, upper, name)
        self.limits = limits
        self.dof = len(limits)

    def forward_kinematics(self, config):
        """Compute the endpoints of the robot given a configuration
        The last endpoint would be used for visualization of the sampling
        """
        raise NotImplementedError

    def get_edges(self):
        """Return the edges of the robot for collision checking"""
        raise NotImplementedError

    def distance(self, config1, config2):
        """Compute the distance between two configurations"""
        raise NotImplementedError

    def interpolate(self, config1, config2, num):
        """Interpolate between two configurations"""
        raise NotImplementedError

    def check_collision(self, config1, config2, map_array):
        """Check colliding with obstacles between two configurations
        First perform an interpolation between the two configurations,
        then check if any of the interpolated configurations hit obstacles.
       
        arguments:
            config1 - configuration 1
            config2 - configuration 2
            map_corners - corners of the map
            obstacles - list of obstacles
            obstacle_edges - list of edges of obstacles, including map edges
        
        return:
            True if colliding with obstacles between the two configurations
        """
        # Get intepolated configurations in between config1 and config2
        configs_between = self.interpolate(config1, config2)
        
        # check if any of these configurations hit obstacles
        for config in configs_between:
            if self.check_collision_config(config, map_array):
                return True
        return False

    def check_collision_config(self, config, map_corners, obstacles, obstacle_edges):
        # Get the endpoint of the robot for collision checking
        robot_endpoint = self.forward_kinematics(config)[-1]

        # Check if the robot endpoint is outside the map
        if not is_in_polygon(robot_endpoint, map_corners):
            return True

        # Check if the robot endpoint is inside any obstacle
        for obstacle in obstacles:
            if is_in_polygon(robot_endpoint, obstacle):
                return True

        # Check for collisions between robot and obstacle edges
        robot_edges = self.get_edges(config)
        
        
        for robot_edge in robot_edges:
            robot_edge = [robot_edge]
            for obstacle_edge in obstacle_edges:
                obstacle_edge = [obstacle_edge]
                if is_intersecting(robot_edge, obstacle_edge):
                    return True

        # If none of the collision checks passed, return False
        return False

    def draw_robot(self, ax, config, edgecolor="b", facecolor="g"):
        """Draw the robot given a configuration on a matplotlib axis.
        This is for visualization purpose only.
        """
        raise NotImplementedError


class KinematicChain(Robot):
    """Kinematic chain robot class
    A planar robot with a fixed base and pure revolute joints.
    Each link is a line segment.
    """

    def __init__(self, link_lengths, base=[0.1, 0.1]):
        """Initialize with a list of link lengths, and a fixed base."""
        self.base = base
        self.link_lengths = link_lengths
        self.num_joints = len(link_lengths)
        # Limits in each dof
        # assume all to be (-pi, pi)
        super().__init__(limits=[
            (-np.pi, np.pi, "r") for _ in range(self.num_joints)
        ])

    def forward_kinematics(self, config):
        """Compute the joint coordinates given a configuration of joint angles.
        The last endpoint would be used for visualization of the sampling
        arguments:
            config: A list of joint angles in radians.

        return:
            edges: A list of joint coordinates.
        """
        # Initialize the starting point as the fixed base
        joint_positions = [self.base]
        start_point = np.array(self.base)
        angle = 0
        # Compute the end points of each joint based on the configuration
        if config is not None:
            for i in range(self.num_joints):
                angle += config[i]
                end_point = start_point + np.array([int(self.link_lengths[i] * np.cos(angle)), int(self.link_lengths[i] * np.sin(angle))])
                joint_positions.append(end_point)
                start_point = end_point
        return joint_positions

    def get_edges(self, config):
        """Compute the link line segments of the robot given a configuration.
        arguments:
            config: A list of joint angles in radians.

        return:
            edges: A list of line segments representing the link line segments.
        """
        # Check configuration length
        assert (
            len(config) == self.num_joints
        ), "Configuration should match the number of joints"

        ### YOUR CODE HERE ###
        edges_list = []

        joint_positions = self.forward_kinematics(config)

        for i in range(len(joint_positions) - 1):
            edges_list.append([joint_positions[i], joint_positions[i + 1]])
        edges = [(tuple(point1), tuple(point2)) for point1, point2 in edges_list]
        return edges

    def distance(self, config1, config2):
        """Calculate the euclidean distance between two configurations
        arguments:
            p1 - config1, [joint1, joint2, joint3, ..., jointn]
            p2 - config2, [joint1, joint2, joint3, ..., jointn]

        return:
            A Euclidean distance in S^1 x S^1 x ... x S^1 space
        """
        
        return np.linalg.norm(np.array(config1) - np.array(config2)) 

    def interpolate(self, config1, config2, num=10):
        """Interpolate between two configurations
        arguments:
            p1 - config1, [joint1, joint2, joint3, ..., jointn]
            p2 - config2, [joint1, joint2, joint3, ..., jointn]

        return:
            A Euclidean distance in 
            list with num number of configs from linear interploation in S^1 x S^1 x ... x S^1 space.
        """
        interpolated_configs = []
        # log.D("Interpolating: "+str(num))
        for i in range(num):
            alpha = i / (num - 1)
            interpolated_config = [(1 - alpha) * c1 + alpha * c2 for c1, c2 in zip(config1, config2)]
            interpolated_configs.append(interpolated_config)
        return interpolated_configs
    
    def is_link_intersecting(self, E1, E2, config):
        joints = self.forward_kinematics(config)
        for i in range(len(joints)):
            if i > 0:
                joints[i] = joints[i].tolist()
        # log.D('Joints: ',joints)
        Line1 = list(E1)
        A,B = Line1
        A1, B1 = A
        A2, B2 = B
        Line2 = list(E2)
        A,B = Line2
        A3, B3 = A
        A4, B4 = B
        Line1 = [[A1,B1],[A2,B2]]
        Line2 = [[A3,B3],[A4,B4]]
        # print('Line1,Line2:',Line1,Line2)
        if A1 == A2:
            A1 = A1-0.0001
        if A3 == A4:
            A3 = A3-0.0001
        
        Point = line_intersection(Line1, Line2)
        if Point:
            if list(Point) not in joints:
                # log.D('Collision')
                return True
        return False
    
    def check_collision_config(self, config, map_array):
        # Get endpoints using forward Kinematics
        # log.D("Checking Collision in Kinematic Chain Class")
        robot_endpoint = self.forward_kinematics(config)[-1]
        # log.D("Current config:"+str(config)+"Current Endpoint:"+str(robot_endpoint))
        # log.D("******")
        # Get all robot edges
        robot_edges = self.get_edges(config)
        # Check if Arm is inside the map
        Joints = self.forward_kinematics(config)
        for Each_Joint in Joints:
            x,y = Each_Joint
            if x<0 or x>899 or y<0 or y>899:
                log.D("Robot Out of Bounds")
                return True
            if map_array[int(x),int(y)] == 0:
                log.D("Current config: "+str(x)+","+str(y)+"_____Collision!")
                return True
        # Check if all robot edges collide
        for edge in robot_edges:
            if self.Arm_Collision_Check(edge,map_array):
                log.D('Obstacle Collison!')
                return True
            for temp in robot_edges:
                if temp != edge:
                    if self.is_link_intersecting(temp, edge, config):
                        # print('R:',edge)
                        # print('T:',temp)
                        log.D('Self Collison!')
                        return True
        return False
    
    def Arm_Collision_Check(self,line,map_array):
        # log.D('Length of Map_Array: '+str(len(map_array)))
        p1,p2 = line
        # print("Line: ",line)
        point_test = [0,0]
        point_test[1],point_test[0] = p1
        p1 = [point_test[0],point_test[1]]
        point_test[1],point_test[0] = p2
        p2 = [point_test[0],point_test[1]]
        # log.D(p1)
        # log.D(p2)
        MP = []
        MP.append(p1)
        MP.append(p2)
        Weight = self.point_distance(p1,p2)
        No_of_Mp = int(Weight//1)                                                       #No of midpoints to be taken is dependent on length of edge
        No_of_Mp = No_of_Mp*2
        # print('Number of Midpoints: ', No_of_Mp)
        temp_List = MP
        Collision_Flag = False
        Done = False
        while Done == False and Collision_Flag == False:
            # print('Beginning of the while loop! lenMP: ',len(MP))
            for temp in range(len(MP)-1):
                Px = int((MP[temp][0]+MP[temp+1][0])/2)
                Py = int((MP[temp][1]+MP[temp+1][1])/2)
                # print(Px,Py)
                if map_array[Px,Py] == 1:
                    temp_List.append([Px,Py])
                    temp_List = sorted(temp_List,key = lambda x: x[1])
                    temp_List = sorted(temp_List,key = lambda x: x[0])
                elif map_array[Px,Py] == 0:
                    Collision_Flag = True
                    # print('Collision')
                    return True
            if len(MP) >= No_of_Mp:
                # print('LenMP: ',len(MP),'No of MP',No_of_Mp)
                Done = True
            MP = temp_List
            if Done == True and Collision_Flag ==False:
                # print('No Collision')
                return False
    
    def draw_robot(self, ax, config, edgecolor="b", facecolor="black"):
        # compute joint positions and draw lines
        positions = self.forward_kinematics(config)
        # Draw lines between each joint
        for i in range(len(positions) - 1):
            line = np.array([positions[i], positions[i + 1]])
            ax.plot(line[:, 0], line[:, 1], color=edgecolor)
        # Draw joint
        for i in range(len(positions)):
            ax.scatter(positions[i][0], positions[i][1], s=5, c=facecolor)
    
    def point_distance(self,p1,p2):                                                                                # Function to calculate distance between two points
        x1, y1 = p1
        x2, y2 = p2
        dx = x1 - x2
        dy = y1 - y2
        return sqrt(dx * dx + dy * dy)

class PointRobot(Robot):
    """2D Point robot class"""

    def __init__(self):
        """Initialize the robot with no limits in x, y (0, 1000))"""
        super().__init__(limits=[
            (0, 1000, "x"),
            (0, 1000, "y")
        ])

    def forward_kinematics(self, config):
        """Simply return the configuration as the endpoint"""
        return [config]

    def get_edges(self, config):
        """Simply return an empty list"""
        return []

    def distance(self, config1, config2):
        """Euclidean distance"""
        x_diff = config1[0] - config2[0]
        y_diff = config1[1] - config2[1]
        return np.sqrt(x_diff**2 + y_diff**2)

    def interpolate(self, config1, config2, num=50):
        """Interpolate between two configurations"""
        configs_between = zip(
            np.linspace(config1[0], config2[0], num),
            np.linspace(config1[1], config2[1], num)
        )
        return configs_between

    def check_collision_config(self, config, map_array):
        # Get the endpoint of the robot for collision checking
        
        # If none of the collision checks passed, return False
        return False

    def draw_robot(self, ax, config, edgecolor="b", facecolor="g"):
        ax.scatter(config[0], config[1], s=20, c=edgecolor)
