import numpy as np
from scipy import spatial

from sampling_method_New import SamplingMethod
from utils import angle_diff, wrap_to_pi

mode = "I"

class log:

    def D(String):
        if mode == "D":
            print('Debug: ',String)
    
    def I(String):
        if mode == "I" or mode == "D":
            print('Info: ',String)
        
    def E(String):
        print('Error: ',String)

class Node:
    """Class for each tree node"""

    def __init__(self, config):
        self.config = config  # configuration
        self.parent = None  # parent node / edge
        self.cost = 0.0  # cost to parent / edge weight
        self.index = 0

class RRT(SamplingMethod):
    """RRT/RRT* class"""

    def __init__(self, sampling_method, n_configs, kdtree_d=10):
        """Specifying number of configs and sampling method to initialize
        arguments:
            sampling_method - name of the chosen sampling method
            n_configs - number of configs to sample
            kdtree_d - the distance of a node to be considered as a neighbor
        """
        super().__init__()
        self.sampling_method = sampling_method
        self.n_configs = n_configs
        self.kdtree_d = kdtree_d

        # kd tree spatial.KDTree([[]]), euclidean distance only
        self.kdtree = None
        self.samples = []  # list of sampled config nodes
        self.solution = []  # list of nodes of the found solution
        self.IRRT_Flag = False

    def init_map(self):
        """Intialize the map before each search"""
        # Add start and goal nodes
        self.start_node = Node(self.start)
        self.goal_node = Node(self.goal)
        # Initialize the containers
        self.samples = []
        self.solution = []

        # Update samples and kdtree
        self.samples.append(self.start_node)
        self.update_kdtree()

    def update_kdtree(self):
        """Update the kd tree after new node is added"""
        self.kdtree = spatial.cKDTree(
            [node.config for node in self.samples]
        )

    def get_new_point(self, goal_bias):
        """Choose the goal or generate a random point in configuration space
        arguments:
            goal_bias - the possibility of choosing the goal
                        instead of a random point

        return:
            point - a new point in the configuration space
        """
        # Select goal
        if np.random.uniform() < goal_bias:
            point = list(self.goal)
        # Or generate a random point
        else:
            point = []
            # sample in each dimension
            for i in range(self.robot.dof):
                point.append(
                    np.random.uniform(
                        self.robot.limits[i][0], 
                        self.robot.limits[i][1]
                    )
                )
        return point

    def get_nearest_node(self, point):
        """Find the nearest node in self.samples from the new point
        arguments:
            point - the new point in configuration space

        return:
            the nearest node in self.samples
        """
        # Use kdtree to find the neighbors within neighbor size
        _, ind = self.kdtree.query(point)
        return self.samples[ind]

    def extend(self, extension_d, goal_bias=0.05):
        """Extend a new node from the current tree
        arguments:
            extension_d - the extension distance
            goal_bias - the possibility of choosing the goal
                        instead of a random sample point

        Create and add a new node if valid.
        """
        # Generate a new point and find the nearest node
        new_point = self.get_new_point(goal_bias)
        nearest_node = self.get_nearest_node(new_point)

        # Calculate new node location by extending the nearest node
        # compute the direction to move
        diff = np.zeros(self.robot.dof)
        for i in range(self.robot.dof):
            # position
            if (self.robot.limits[i][2] != "r"):
                diff[i] = new_point[i] - nearest_node.config[i]
            # rotation
            else:
                # find the shortest angle
                diff[i] = angle_diff(
                    nearest_node.config[i], new_point[i], absolute=False
                )
        # get unit vector 
        if np.linalg.norm(diff) == 0:
            # same configuration
            return None

        new_config = (
            np.array(nearest_node.config) 
            + diff / np.linalg.norm(diff) * extension_d
        ).tolist()
        # wrap the angle if necessary
        for i in range(self.robot.dof):
            if self.robot.limits[i][2] == "r":
                new_config[i] = wrap_to_pi(new_config[i])

        # Check if the new configuration is valid
        if self.check_collision(new_config, nearest_node.config):
            return None
        
        # Create a new node
        new_node = Node(new_config)
        cost = self.robot.distance(new_config, nearest_node.config)
        # add the new node to the tree
        new_node.parent = nearest_node
        new_node.cost = cost
##        log.D('New_Point: '+str(new_node.config))
##        log.D('Length of Samples: '+str(len(self.samples)))
        
        if self.IRRT_Flag == False:
            self.samples.append(new_node)
            self.update_kdtree()
            return new_node
        elif self.IRRT_Flag == True:
            c_i = self.robot.distance(self.start_node.config,self.goal_node.config)*1.2
            # log.D('C_I: '+str(c_i))
            if self.goal_node.parent is None:
                self.samples.append(new_node)
                self.update_kdtree()
                return new_node
            # goal found
            elif self.goal_node.parent is not None:
                # log.D('Path Found!')
                X_Dis = self.robot.distance(self.start_node.config,new_node.config)+self.robot.distance(new_node.config,self.goal_node.config)
                # log.D('X_Dis: '+str(X_Dis))
                if X_Dis > c_i:
                    return None
                elif X_Dis <= c_i:
                    # log.D('Here')
                    self.samples.append(new_node)
                    self.update_kdtree()
                    return new_node

    def connect_to_near_goal(self, new_node):
        """Check if the new node is near and has a valid path to goal node
        If yes, connect the new node to the goal node
        """
        # Check if goal is close and there is a valid path
        dis = self.robot.distance(new_node.config, self.goal)
        if (
            dis < self.kdtree_d 
            and not self.check_collision(new_node.config, self.goal)
        ):
            # connect directly to the goal
            self.goal_node.cost = dis
            self.goal_node.parent = new_node
            self.samples.append(self.goal_node)
            self.update_kdtree()

    def get_path_and_cost(self, start_node, end_node):
        """Compute path cost starting from a start node to an end node
        arguments:
            start_node - path start node
            end_node - path end node

        return:
            cost - path cost
        """
        cost = 0
        curr_node = end_node
        path = [curr_node]

        # Keep tracing back
        # until finding the start_node
        # or no path exists
        while curr_node != start_node:
            cost += curr_node.cost
            parent = curr_node.parent
            if parent is None:
                print("There is no path from the given start to goal")
                return [], 0

            curr_node = parent
            path.append(curr_node)

        return path[::-1], cost

    def get_neighbors(self, new_node):
        """Get the neighbors within the neighbor distance from the node
        arguments:
            new_node - a new node
            size - the neighbor size

        return:
            neighbors - list of neighbors within the neighbor distance
        """
        ### YOUR CODE HERE ###
        New_Neighbours = []
        Temp, Neighbours = self.kdtree.query(new_node.config,k = 10,distance_upper_bound = 30)
        for i in Neighbours:
            if i < len(self.samples):
                if self.samples[i].config is not new_node.config:
                    New_Neighbours.append(i)
                if self.samples[i].config is new_node.config:
                    new_node.index = i
                    # log.D('New_Node_Index: '+str(new_node.index))
        return New_Neighbours

    def rewire(self, new_node, neighbors):
        """Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors of the new node

        Rewire the new node if connecting to a new neighbor node
        will give least cost.
        Rewire all the other neighbor nodes.
        """
        path0,cost0 = self.get_path_and_cost(self.start_node, new_node)
        # log.D('Current Node: '+str(self.samples[new_node.index].config)+', Parent: '+str(self.samples[new_node.index].parent.config)+', Cost: '+str(cost0))
        ### YOUR CODE HERE ###
        
        ### Checking and updating parents for new_node ###
        for i in neighbors: 
            path,cost = self.get_path_and_cost(self.start_node, self.samples[i])
            cost1 = self.robot.distance(self.samples[i].config, self.samples[new_node.index].config)
            total_cost = cost + cost1
            path0,cost0 = self.get_path_and_cost(self.start_node, self.samples[new_node.index])
            if total_cost < cost0:
                if self.check_collision(self.samples[new_node.index].config, self.samples[i].config) == False:
                    self.samples[new_node.index].parent = self.samples[i]
                    self.samples[new_node.index].cost = self.robot.distance(self.samples[new_node.index].config, self.samples[new_node.index].parent.config)
                    path0,cost0 = self.get_path_and_cost(self.start_node, self.samples[new_node.index])
        path,new_cost = self.get_path_and_cost(self.start_node, self.samples[new_node.index])
        
        ### Checking and updating parents for neighbors of new_node ###
        for i in neighbors:
            if self.samples[i] is not self.samples[new_node.index].parent:
                path,cost = self.get_path_and_cost(self.start_node, self.samples[i])
                total_cost = new_cost + self.robot.distance(self.samples[new_node.index].config, self.samples[i].config)
                if total_cost < cost:
                    if self.check_collision(self.samples[new_node.index].config, self.samples[i].config) == False:
                        self.samples[i].parent = self.samples[new_node.index]
                        self.samples[i].cost = self.robot.distance(self.samples[new_node.index].config, self.samples[i].config) 

    def RRT(self):
        """RRT search function
        In each step, extend a new node if possible,
        and check if reached the goal
        """
        # Start searching
        while len(self.samples) < self.n_configs:
            # Extend a new node until
            # all the points are sampled
            # or find a path
            new_node = self.extend(extension_d=self.kdtree_d, goal_bias=0.05)

            # If goal is not found, try to connect new node to goal
            if self.goal_node.parent is None:
                if new_node is not None:
                    self.connect_to_near_goal(new_node)
            # goal found
            else:
                break

        # Output
        if self.goal_node.parent is not None:
            num_nodes = len(self.samples)
            path, length = self.get_path_and_cost(
                self.start_node, self.goal_node
            )
            self.solution = [node.config for node in path]
            print("The constructed tree has %d of nodes" % num_nodes)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        return self.solution

    def RRT_star(self):
        """RRT* search function
        In each step, extend a new node if possible,
        and rewire the node and its neighbors.
        """
        # Start searching
        while len(self.samples) < self.n_configs:
            # Extend a new node
            new_node = self.extend(extension_d=self.kdtree_d, goal_bias=0.05)

            # Rewire the new node and its neighbors
            if new_node is not None:
                neighbors = self.get_neighbors(new_node)
                self.rewire(new_node, neighbors)

            # If goal is not found, try to connect new node to goal
            if self.goal_node.parent is None and new_node is not None:
                self.connect_to_near_goal(new_node)
            log.I('No. Of Samples: '+str(len(self.samples)))
        # Output
        if self.goal_node.parent is not None:
            num_nodes = len(self.samples)
            path, length = self.get_path_and_cost(
                self.start_node, self.goal_node
            )
            self.solution = [node.config for node in path]
            log.D('Solution: '+str(self.solution))
            print("The constructed tree has %d of nodes" % num_nodes)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        return self.solution

    def Informed_RRT_star(self):
        '''Informed RRT* search function
        In each step, extend a new node if possible, and rewire the node and its neighbors
        Once a path is found, an ellipsoid will be defined to constrained the sampling area
        '''
        ### YOUR CODE HERE ###
        self.IRRT_Flag = True
        while len(self.samples) < self.n_configs:
            new_node = self.extend(extension_d=self.kdtree_d, goal_bias=0.05)
            
            if new_node is not None:
                neighbors = self.get_neighbors(new_node)
                self.rewire(new_node, neighbors)
            
            if self.goal_node.parent is None:
                if new_node is not None:
                    self.connect_to_near_goal(new_node)
            log.I('No. Of Samples: '+str(len(self.samples)))
        
        self.IRRT_Flag = False
        # Output
        if self.goal_node.parent is not None:
            num_nodes = len(self.samples)
            path, length = self.get_path_and_cost(
                self.start_node, self.goal_node
            )
            self.solution = [node.config for node in path]
            log.D('Solution: '+str(self.solution))
            print("The constructed tree has %d of nodes" % num_nodes)
            print("The path length is %.2f" % length)
        else:
            print("No path found")
        
        return self.solution


    def plan(self, start, goal):
        """Search for a path in graph given start and goal location

        arguments:
            start - start configuration
            goal - goal configuration
        """
        self.start = start
        self.goal = goal
        self.init_map()

        if self.sampling_method == "RRT":
            self.solution = self.RRT()
        elif self.sampling_method == "RRT_star":
            self.solution = self.RRT_star()
        elif self.sampling_method == "Informed_RRT_star":
            self.solution = self.Informed_RRT_star()
        else: 
            raise ValueError(f"Sampling method:{self.sampling_method} does not exist!")

        return self.solution

    def visualize_sampling_result(self, ax):
        """ Visualization the sampling result."""
        # Draw Trees / sample points
        for node in self.samples:
            # node
            pos1 = self.robot.forward_kinematics(node.config)[-1]
            ax.plot(
                pos1[0], 
                pos1[1], 
                markersize=5, 
                marker=".", 
                color="g", 
                alpha=0.3
            )
            # edge
            if node == self.start_node:
                continue
            pos2 = self.robot.forward_kinematics(node.parent.config)[-1]
            ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color="y",
                alpha=0.3
            )
