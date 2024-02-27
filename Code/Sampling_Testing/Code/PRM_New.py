import math

# from klampt import WorldModel
# from klampt import vis
# from klampt.math import se3
# from klampt.model import ik

from math import atan2, degrees
import copy
import numpy as np
import networkx as nx
from scipy import spatial
from RRT_New import log
from utils import interpolate_angle
from sampling_method_New import SamplingMethod
import matplotlib.pyplot as plt


class PRM(SamplingMethod):
    """Probabilistic Roadmap (PRM) planner"""

    def __init__(self, sampling_method="uniform", n_configs=1000, kdtree_d=400):
        """Specifying number of configs and sampling method to initialize
        arguments:
            sampling_method - name of the chosen sampling method
            n_configs - number of configs to sample
            kdtree_d - the distance for kdtree to search for nearest neighbors
        """
        super().__init__()
        self.sampling_method = sampling_method
        self.n_configs = n_configs
        self.kdtree_d = kdtree_d

        # kd tree spatial.KDTree([[]]), euclidean distance only
        self.kdtree = None
        self.samples = []  # list of sampled configs
        self.Final_Samples = []
        self.graph = nx.Graph()  # constructed graph
        self.solution = []  # list of nodes of the found solution
        self.counter0 = 0
        self.counter1 = 0
        self.counter2 = 0
        self.counter3 = 0
        self.counter4 = 0
        self.Num_Samples0 = 10
        self.Num_Samples1 = 38
        self.Num_Samples2 = 38
        self.Num_Samples3 = 0
        self.Num_Samples4 = 19
        self.Klampt_robot_ee = 0
        self.klampt_joint_limits = []

    """
    Section used for inverse kinematics

    # def setup_klampt(self):
    #     # world = WorldModel()
    #     # world.loadElement("D:\WPI\Sem1\Motion_Planning\Final_Project\Code\planar_5R.rob")
    #     # klampt_robot = world.robot(0)
    #     # self.Klampt_robot_ee = klampt_robot.link(klampt_robot.numLinks()-1)
    #     # num_joints = 5
    #     # joint_limit = [-3.14, 3.14]
    #     # self.klampt_joint_limits = [[limit] * num_joints for limit in joint_limit]
    #     # print("Returning robot")
    #     # q0 = klampt_robot.getConfig()
    #     # print(q0)
    #     # return klampt_robot
    #     pass
    #     # Add links to the robot
    #     # link_lengths = [1, 1, 1, 1, 1]

    # def inverse_kinematics(self, robot, target_position):
    #     print("Starting IK")
    #     world = WorldModel()
    #     world.loadElement("D:\WPI\Sem1\Motion_Planning\Final_Project\Code\planar_5R.rob")
    #     klampt_robot = world.robot(0)
    #     self.Klampt_robot_ee = klampt_robot.link(klampt_robot.numLinks()-1)
    #     num_joints = 5
    #     joint_limit = [-3.14, 3.14]
    #     self.klampt_joint_limits = [[limit] * num_joints for limit in joint_limit]
    #     print("Returning robot")
    #     q0 = klampt_robot.getConfig()
    #     for i in range(len(target_position)):
    #         target_position[i] = target_position[i]/120
    #     print("Target Position:",target_position)
    #     klampt_robot.setConfig(q0)
    #     obj = ik.objective(
    #                 self.Klampt_robot_ee,
    #                 local=[0, 0, 0],
    #                 world=[target_position[0], 0, -target_position[1]]
    #                 # In Klampt, the plane is x, (-z)
    #             )
    #     solver = ik.solver(obj)
    #     solver.setJointLimits(self.klampt_joint_limits[0], self.klampt_joint_limits[1])
    #     solver.setMaxIters(100)
    #     solver.setTolerance(1e-3)
    #     success = solver.solve()
    #     print(klampt_robot.getConfig())
    #     return klampt_robot.getConfig()
    """

    def get_sample(self):
        """Get a sample configuration"""
        # Generate a random config
        
        x = -1
        y = -1
        while True:
            sample = []
            for i in range(self.robot.dof):
                sample.append(
                    np.random.uniform(
                        self.robot.limits[i][0], 
                        self.robot.limits[i][1]
                    )
                )
            x,y =  self.robot.forward_kinematics(sample)[-1]
            if x>0 and y>0:
                break
        sample=np.array(sample)
        return sample

    def get_gaussian_offset_sample(self, sample, scale=10):
        """Get another sample at some distance from the given sample
        arguments:
            sample - the given sample
            scale - the standard deviation of the gaussian distribution
                    is defined as limit / scale

        return:
            a sample at some distance from the given sample
        """
        diff = []
        # Gaussian sample in each dimension 
        # with mean 0 and std limit / scale
        for i in range(self.robot.dof):
            diff.append(
                np.random.normal(
                    0.0, 
                    np.abs(
                        self.robot.limits[i][1] - self.robot.limits[i][0]
                    ) / scale
                )
            )
        return sample + np.array(diff)
        # for i in range(self.robot.dof):
        #     if sum_angle<0.349:
        #         diff.append(
        #             np.random.normal(0.0, np.abs(self.robot.limits[i][1] - self.robot.limits[i][0]) / (scale)))
        #         sum_angle = sum_angle + diff[-1]
        #     else:
        #         diff.append(0.0)

    def uniform_sample(self, n_configs):
        """Use uniform sampling and store valid configs
        arguments:
            n_configs - number of configs to sample

        check collision and store valide configs in self.samples
        """
        Timeout_Counter = 0
        dummy_samples = []
        # print("In uniform Sample Method")
        while len(dummy_samples) < n_configs:
            # Generate a uniform random config
            # print("Inside While Loop")
            # Timeout_Counter = Timeout_Counter + 1
            # if Timeout_Counter > 500 and len(dummy_samples) < 1:
            #     print("Breaking!")
            #     break
            sample = self.get_sample()
            # Check obstacle
            if not self.check_collision_config(sample):
                x,y = self.robot.forward_kinematics(sample)[-1]
                # print('x,y:',x,y)
                if x>=0 and y>=0: 
                    # print('Appended')
                    dummy_samples.append(sample)
                    log.I('Number of valid Samples: '+str(len(dummy_samples)))
        self.samples.extend(dummy_samples)
        print(self.samples)
    
    def gaussian_sample(self, n_configs, prob=0.05):
        """Use gaussian sampling and store valid configs
        arguments:
            n_configs - number of configs to sample
            prob - probability of sampling a config with uniform sampling,
                   to avoid long sampling time in certain maps

        check collision and store valide configs in self.samples
        """
        dummy_samples = []
        while len(dummy_samples) < n_configs:
            # Generate a uniform random config
            sample = self.get_sample()

            # Use uniform sampling with probability prob
            if np.random.uniform() < prob:
                # Check obstacle
                if not self.check_collision_config(sample):
                    dummy_samples.append(sample)
                continue

            # Use gaussian sampling
            # Generate a new random config at some distance
            sample2 = self.get_gaussian_offset_sample(sample)

            # Check if the config is close to an obstacle, i.e.,
            # one of them is obstacle and the other is free space
            collision1 = self.check_collision_config(sample)
            collision2 = self.check_collision_config(sample2)
            if (collision1 and not collision2):
                dummy_samples.append(sample2)
            elif (collision2 and not collision1):
                dummy_samples.append(sample)
        self.samples.extend(dummy_samples)

    def bridge_sample(self, n_configs, prob=0.05):
        """Use bridge sampling and store valid configs
        arguments:
            n_configs - number of configs to sample
            prob - probability of sampling a config with uniform sampling,
                   to avoid long sampling time in certain maps

        check collision and store valide configs in self.samples
        """
        dummy_samples = []
        while len(dummy_samples) < n_configs:
            # Generate a uniform random config
            sample = self.get_sample()
            if sample is not None:
            # Use uniform sampling with probability prob
                if np.random.uniform() < prob:
                    # Check obstacle
                    if not self.check_collision_config(sample):
                        dummy_samples.append(sample)
                    continue

                # Use gaussian sampling
                # Generate a new random config at some distance
                sample2 = self.get_gaussian_offset_sample(sample)

                # Check if it is the "bridge" form, i.e.,
                # the center is not obstacle but both ends are obstacles
                # Both are obstacles
                collision1 = self.check_collision_config(sample)
                collision2 = self.check_collision_config(sample2)
                if (collision1 and collision2):
                    # The mid-sample is not an obstacle
                    mid_sample = sample
                    for i in range(self.robot.dof):
                        # position
                        if (self.robot.limits[i][2] != "r"):
                            mid_sample[i] = np.mean([sample[i], sample2[i]])
                        # rotation
                        else:
                            mid_sample[i] = interpolate_angle(
                                sample[i], sample2[i], 3
                            )[1]
                    if (not self.check_collision_config(mid_sample)):
                        dummy_samples.append(mid_sample)
        self.samples.extend(dummy_samples)
                     
    def custom_chain(self, n_configs):
        """
        Custom Sampling method for Kinematic chain.
        
        n_configs: Number of samples required by this sampling method.
        Updates self.samples.
        """
        
        dummy_samples = []
        while len(dummy_samples) < n_configs:
            # Generate a uniform random config
            sample = self.get_sample()
            if sample is not None:
                if not self.check_collision_config(sample):
                    dummy_samples.append(sample)
                else:
                    dummy_samples.extend(self.get_intermediate_custom_chain(sample))
        
        self.samples.extend(dummy_samples)

    def get_intermediate_custom_chain(self, sample):
        """
        Get Custom non colliding Samples for colliding Uniform Sample.

        Sample: The colliding uniform sample
        Returns A list of non colliding samples derived from 'Sample'
        """
        
        Temp_Node_List = []
        end_point = self.robot.forward_kinematics(sample)[-1]
        Angle = atan2(end_point[1],end_point[0])
        Return_Node_List = []
        Deviation = -0.06
        Temp_Node = sample[:]
        Val = sample[0]
        for i in range(7):
            if Deviation == 0:
                Deviation += 0.02
                continue
            Temp_Node[0] = Val + Deviation
            Deviation += 0.02
            Temp_Node_List.append(copy.deepcopy(Temp_Node))

        offset = [0.9, 1, 1.2, 1.4, 1.6]
        for OS in offset:
            Node1 = [Angle+OS,-(2*OS),OS,0,0]
            Node2 = [Angle+OS,-(2*OS),0,(2*OS),-OS]
            Temp_Node_List.append(Node1)
            Temp_Node_List.append(Node2)
        
        for MyNode in Temp_Node_List:
            if not self.check_collision_config(MyNode):
                Return_Node_List.append(MyNode)
        
        return Return_Node_List

    def add_vertices_pairs(self, pairs):
        """Add pairs of vertices to graph as weighted edge
        arguments:
            pairs - pairs of vertices of the graph

        check collision, compute weight and add valide edges to self.graph
        """
        for pair in pairs:
            if pair[0] == "start":
                config1 = self.samples[-2]
                log.D("Start config: "+str(config1))
                log.D("Adding Start Pairs")
            elif pair[0] == "goal":
                config1 = self.samples[-1]
                log.D("Goal config: "+str(config1))
                log.D("Adding Goal Pairs")
            else:
                config1 = self.samples[pair[0]]
            config2 = self.samples[pair[1]]
            log.D("______________________________________________")
            log.D("______________________________________________")
            log.D("Checking Pair: "+str(config1)+","+str(config2))
            if config1 is not config2:
                if not self.check_collision(config1, config2):
                    d = self.robot.distance(config1, config2)
                    self.graph.add_weighted_edges_from([(pair[0], pair[1], d)])
                    log.D('Edge Added!')
            log.D("______________________________________________")
            log.D("______________________________________________")
        log.D("Edges:"+str(self.graph.edges.data()))
        
    def add_Final_vertices_pairs(self, pairs):
        i=0
        for pair in pairs:
            log.I("Checking pair "+str(i)+" out of total "+str(len(pairs))+" Pairs")
            if pair[0] == "start":
                config1 = self.Final_Samples[-2]
                log.D("Start config: "+str(config1))
                log.D("Adding Start Pairs")
            elif pair[0] == "goal":
                config1 = self.Final_Samples[-1]
                log.D("Goal config: "+str(config1))
                log.D("Adding Goal Pairs")
            else:
                config1 = self.Final_Samples[pair[0]]
            config2 = self.Final_Samples[pair[1]]
            log.D("______________________________________________")
            log.D("______________________________________________")
            log.D("Checking Pair: "+str(config1)+","+str(config2))
            if config1 is not config2:
                if not self.check_collision(config1, config2):
                    d = self.robot.distance(config1, config2)
                    self.graph.add_weighted_edges_from([(pair[0], pair[1], d)])
                    log.I('Edge Added!')
            log.D("______________________________________________")
            log.D("______________________________________________")
            i += 1
        log.D("Edges:"+str(self.graph.edges.data()))
    
    def connect_vertices(self):
        """Add nodes and edges to the graph from sampled configs
        arguments:
            kdtree_d - the distance for kdtree to search for nearest neighbors

        Add nodes to graph from self.samples
        Build kdtree to find neighbor pairs and add them to graph as edges
        """
        # Finds k nearest neighbors
        # kdtree
        self.kdtree = spatial.cKDTree(self.samples)
        pairs = self.kdtree.query_pairs(self.kdtree_d)
        log.I("Connecting Vertices")
        
        # Add the neighbor to graph
        # self.graph.add_nodes_from(range(len(self.samples)))
        self.add_vertices_pairs(pairs)
        
    def connect_Final_vertices(self):
        
        print("Final Samples:",self.Final_Samples)
        self.kdtree = spatial.cKDTree(self.Final_Samples)
        pairs = self.kdtree.query_pairs(self.kdtree_d)
        log.I("Connecting Final Vertices")
        
        # Add the neighbor to graph
        self.graph.add_nodes_from(range(len(self.Final_Samples)))
        self.add_Final_vertices_pairs(pairs)

    def sample(self):
        """Construct a graph for PRM
        arguments:
            n_configs - number of configs try to sample,
                    not the number of final sampled configs
            sampling_method - name of the chosen sampling method
            kdtree_d - the distance for kdtree to search for nearest neighbors

        Sample configs, connect, and add nodes and edges to self.graph
        """
        # Initialization

        self.path = []
        self.graph.clear()
        # Sample using given methods
######################################################################
        log.I("Getting New Samples")
        if self.sampling_method == "uniform":
            self.uniform_sample(10)
        elif self.sampling_method == "gaussian":
            self.gaussian_sample(10)
        elif self.sampling_method == "bridge":
            self.bridge_sample(20)
        elif self.sampling_method == "custom_chain":
            self.custom_chain(10)
######################################################################
        self.connect_vertices()
        
    def Segment_Samples(self):
        """
        Modifies the uniform samples according to requirement for each segment of  the map.
        """
        log.I("Modifying samples for Segments!")
        
        for sample in self.samples:
            endpoint =  self.robot.forward_kinematics(sample)[-1]
            if endpoint[0] == self.robot.base[0] and endpoint[1] == self.robot.base[1]:
                continue
            x,y = endpoint
            if x>0 and x<300 and y>0 and y< 300:
                method = self.sampling_method[0]
                if self.counter0 < self.Num_Samples0:
                    Temp, count = self.Update_Segment_Sample(sample, method, 0, 300, 0, 300)
                    if Temp is True:
                        self.counter0 = self.counter0 + count
            if x>300 and x<600 and y>0 and y< 300:
                method = self.sampling_method[1]
                if self.counter1 < self.Num_Samples1:
                    Temp, count = self.Update_Segment_Sample(sample, method, 300, 600, 0, 300)
                    if Temp is True:
                        self.counter1 = self.counter1 + count
            if x>0 and x<300 and y>300 and y<600:
                method = self.sampling_method[2]
                if self.counter2 < self.Num_Samples2:
                    Temp, count = self.Update_Segment_Sample(sample, method, 0, 300, 300, 600)
                    if Temp is True:
                        self.counter2 = self.counter2 + count
            if x>600 and x<900 and y>0 and y<300:
                method = self.sampling_method[3]
                if self.counter3 < self.Num_Samples3:
                    Temp, count = self.Update_Segment_Sample(sample, method, 600, 900, 0, 300)
                    if Temp is True:
                        self.counter3 = self.counter3 + count
            if x>300 and x<600 and y>300 and y<600:
                method = self.sampling_method[4]
                if self.counter4 < self.Num_Samples4:
                    Temp, count = self.Update_Segment_Sample(sample, method, 300, 600, 300, 600)
                    if Temp is True:
                        self.counter4 = self.counter4 + count
                        

            if self.counter0 >= self.Num_Samples0 and self.counter1 >= self.Num_Samples1 and self.counter2 >= self.Num_Samples2 and self.counter3 >= self.Num_Samples3 and self.counter4 >= self.Num_Samples4:
                break
            log.I("----------------------")
            log.I("0: "+str(self.counter0))
            log.I("1: "+str(self.counter1))
            log.I("2: "+str(self.counter2))
            log.I("4: "+str(self.counter4))
            log.I("----------------------")
 
    def Update_Segment_Sample(self, sample, method, xmin, xmax, ymin, ymax):
        """
        Updates Samples based on Sampling method for each segment.
        
        sample - Sample to be modified
        method - Sampling method to be used
        xmin, xmax - Boundary values for each segment on x-axis
        ymin, ymax - Boundary values for each segment on y-axis
        
        """

        if method == "uniform":
            print("Checking Uniform")
            if sample is not None:
                count = 0
                collision1 = self.check_collision_config(sample)
                if not collision1:
                    self.Final_Samples.append(sample)
                    count = 1
                    return True, count
            return False, 0

        elif method == "gaussian":
            count = 0
            if np.random.uniform() < 0.05:
                # Check obstacle
                if not self.check_collision_config(sample):
                    self.Final_Samples.append(sample)
                    count += 1
                    print("Gaussian Appended")
                    return True, count
            
            print("Checking Gaussian")
            Flag = False
            collision1 = self.check_collision_config(sample)
            if collision1:
                Gaussian_Timer_Counter = 0
                while Flag == False:
                    if Gaussian_Timer_Counter > 500:
                        break
                    Gaussian_Timer_Counter = Gaussian_Timer_Counter + 1
                    sample2 = self.get_gaussian_offset_sample(sample)
                    sample_endpoint = self.robot.forward_kinematics(sample2)[-1]
                    if sample_endpoint[0]>xmin and sample_endpoint[0]<xmax and sample_endpoint[1]>ymin and sample_endpoint[1]<ymax:
                        if sample2 is not None:
                            collision2 = self.check_collision_config(sample2)
                            if collision2 == False:
                                Flag = True
                                self.Final_Samples.append(sample2)
                                count += 1
                                print("Gaussian Appended")
                                return True, count
                            
            if not collision1:
                count = 0
                Gaussian_Timer_Counter = 0
                while Flag == False:
                    if Gaussian_Timer_Counter > 500:
                        break
                    Gaussian_Timer_Counter = Gaussian_Timer_Counter + 1
                    sample2 = self.get_gaussian_offset_sample(sample)
                    sample_endpoint = self.robot.forward_kinematics(sample2)[-1]
                    if sample_endpoint[0]>xmin and sample_endpoint[0]<xmax and sample_endpoint[1]>ymin and sample_endpoint[1]<ymax:
                        if sample2 is not None:
                            collision2 = self.check_collision_config(sample2)
                            if collision2 == True:
                                Flag = True
                                self.Final_Samples.append(sample2)
                                count += 1
                                print("Gaussian Appended")
                                return True, count
            return False, 0

        elif method == "bridge":
            count = 0
            if np.random.uniform() < 0.05:
                # Check obstacle
                if not self.check_collision_config(sample):
                    self.Final_Samples.append(sample)
                    count += 1
                    print("Bridge Appended")
                    return True, count
            print("Checking Bridge")
            sample2 = self.get_gaussian_offset_sample(sample) 
            sample_endpoint = self.robot.forward_kinematics(sample2)[-1]
            if sample_endpoint[0]>xmin and sample_endpoint[0]<xmax and sample_endpoint[1]>ymin and sample_endpoint[1]<ymax:
                collision1 = self.check_collision_config(sample)
                collision2 = self.check_collision_config(sample2)
                if (collision1 and collision2):
                    # The mid-sample is not an obstacle
                    mid_sample = sample
                    for i in range(self.robot.dof):
                        mid_sample[i] = interpolate_angle(sample[i], sample2[i], 3)[1]
                    if (not self.check_collision_config(mid_sample)):
                        if mid_sample is not None:
                            self.Final_Samples.append(mid_sample)
                            count += 1
                            print("Bridge Appended")
                            return True, count
            return False, 0
        
        elif method == "custom_chain":
            print("Checking Custom Chain")
            count = 0
            if np.random.uniform() < 0.05:
                # Check obstacle
                if not self.check_collision_config(sample):
                    self.Final_Samples.append(sample)
                    count += 1
                    print("Custom Appended")
                    return True, count
            collision1 = self.check_collision_config(sample)
            if not collision1:
                self.Final_Samples.append(sample)
                count += 1
                return True, count
            elif collision1:
                Temp_List = self.get_intermediate_custom_chain(sample)
                for temp_sample in Temp_List:
                    sample_endpoint = self.robot.forward_kinematics(temp_sample)[-1]
                    if sample_endpoint[0]>xmin and sample_endpoint[0]<xmax and sample_endpoint[1]>ymin and sample_endpoint[1]<ymax:
                        self.Final_Samples.append(temp_sample)
                        count += 1
                return True, count
            return False, 0
        
    def plan(self, start, goal, redo_sampling=True):
        """Search for a path in graph given start and goal location
        Temporary add start and goal node,
        edges of them and their nearest neighbors to graph.
        Use graph search algorithm to search for a path.

        arguments:
            start - start configuration
            goal - goal configuration
            redo_sampling - whether to rerun the sampling phase
        """
        self.samples = []
        self.solution = []
        length = 0
        dummy1 = []
        dummy1.append(start)
        dummy1.append(goal)
        Array1 = ["start","goal"]
        while length == 0:
        # Sampling and building the roadmap
            if redo_sampling:
                self.sample()
            ############################################
            # fig, ax = plt.subplots()
            # img = 255 * np.dstack((self.map, self.map, self.map))
            # # img = np.flipud(img)
            # ax.imshow(img)
            # node_names = list(range(len(self.samples)))
            # end_points = [
            #     self.robot.forward_kinematics(config)[-1] 
            #     for config in self.samples
            # ]
            # positions = dict(zip(node_names, end_points))
            # nx.draw_networkx_nodes(self.graph,positions,node_size=5)
            
            ###############################################
            dummy = dict(zip(range(len(self.samples)), self.samples))
            self.graph.add_nodes_from(dummy)
            self.graph.add_nodes_from(Array1)
            self.samples.extend(dummy1)
            start_goal_tree = spatial.cKDTree([start, goal])
            neighbors = start_goal_tree.query_ball_tree(self.kdtree, 2*self.kdtree_d)
            start_pairs = [["start", neighbor] for neighbor in neighbors[0]]
            goal_pairs = [["goal", neighbor] for neighbor in neighbors[1]]

            # Add the edge to graph
            self.add_vertices_pairs(start_pairs)
            self.add_vertices_pairs(goal_pairs)

            # Seach using Dijkstra
            found = False
            length = 0
            try:
                self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                    self.graph, "start", "goal"
                )
                length = (
                    nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
                        self.graph, "start", "goal"
                    )
                )
                found = True
                num_nodes = self.graph.number_of_nodes()
                log.I("The constructed graph has %d of nodes" % num_nodes)
                log.I("The path length is %.2f" % length)
            except nx.exception.NetworkXNoPath:
                log.I("No path found")
                
                # plt.axis("on")
                # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                # ax.invert_yaxis()
                # plt.show()
                # log.I("Samples:")
                # for sample in self.samples:
                #     endpoint = self.robot.forward_kinematics(sample)[-1]
                #     print(endpoint)
                # log.I("")
                # log.I("Edges: "+str(self.graph.edges.data()))
            # Remove start and goal node
            self.samples.pop(-1)
            self.samples.pop(-1)
        
        
        # self.graph.remove_nodes_from(["start", "goal"])
        # self.graph.remove_edges_from(start_pairs)
        # self.graph.remove_edges_from(goal_pairs)
        Number_of_Nodes = self.graph.number_of_nodes() - 2
        length = round(length,2)
        
            
        # Final solution
        if found:
            self.solution = [self.samples[i] for i in self.path[1:-1]]
            self.solution.insert(0, start)
            self.solution.append(goal)
        log.D("Solution: "+str(self.solution))
                
        return self.solution, length, Number_of_Nodes
    
    def test_plan(self, start, goal, redo_sampling=True):
        self.samples = []
        self.solution = []
        length = 0
        dummy1 = []
        dummy1.append(start)
        dummy1.append(goal)

        ############# Remove this (Inverse Kinematics Implementation) #####################
        # Points = []
        # Angle_points = []
        # k_robot = self.setup_klampt()
        # # for temp in range(10):
        # while len(Angle_points)<20:    
        #     Coord = []
        #     Coord.append(int(np.random.uniform(150, 450)))
        #     Coord.append(int(np.random.uniform(0, 300)))
        #     print("Point before IK:",Coord)
        #     Angle_points.append(self.inverse_kinematics(k_robot,Coord))
        #     print("Angle points:",Angle_points)
        #     print(self.robot.forward_kinematics(Angle_points[-1])[-1])
        #     print("Point after IK",[int(self.robot.forward_kinematics(Angle_points[-1])[-1][0]),int(self.robot.forward_kinematics(Angle_points[-1])[-1][1])])
        #     if (self.check_collision_config(Angle_points[-1])):
        #         Angle_points.pop(-1)
        #         Coord.pop(-1)
        #     print("Length:",len(Angle_points))
        # Angle_points = [(1.3, -2.5, 1.8, -1.5, 0.8),(1.57, -2, 0.5, 0.6, 0.5)]
        # dummy = dict(zip(range(len(Angle_points)), Angle_points))
        # self.graph.add_nodes_from(dummy)
        # fig, ax = plt.subplots()
        # img = 255 * np.dstack((self.map, self.map, self.map))
        # img = np.flipud(img)
        # ax.imshow(img)
        # node_names = list(range(len(Angle_points)))
        # end_points = [[int(self.robot.forward_kinematics(config)[-1][0]),int(self.robot.forward_kinematics(config)[-1][1])] for config in Angle_points]
        # positions = dict(zip(node_names, end_points))
        # for i in Angle_points:
        #     # draw robot
        #     self.robot.draw_robot(ax, i, edgecolor="black")
        # print("Graph Nodes:",self.graph.nodes(data=True))
        # nx.draw_networkx_nodes(self.graph,positions,node_size=5)
        # # nx.draw_networkx_nodes(self.graph,positions2,node_size=5,node_color="g")
        # ax.invert_yaxis()
        # plt.axis("on")
        # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()
        ############# Remove this #####################


        NS0 = 10
        NS1 = 38
        NS2 = 38
        NS3 = 10
        NS4 = 19
        
        i = 4

        while length == 0:
            i += 1
            

            self.Num_Samples0 = NS0 * i
            self.Num_Samples1 = NS1 * i
            self.Num_Samples2 = NS2 * i
            self.Num_Samples3 = NS3 * i
            self.Num_Samples4 = NS4 * i
            
            # Sampling and building the roadmap
            
            log.I("Getting Random Samples")
            for t in range(50000):
                self.samples.append(self.get_sample())
            
            self.Segment_Samples()
            end_points = [self.robot.forward_kinematics(config)[-1] for config in self.Final_Samples]
           
            self.graph.clear()
            # dummy = dict(zip(range(len(self.Final_Samples)), self.Final_Samples))
            # self.graph.add_nodes_from(dummy)
            self.connect_Final_vertices()
            
            # ############################################
            # # Plot to see endpoints of samples:

            # fig, ax = plt.subplots()
            # img = 255 * np.dstack((self.map, self.map, self.map))
            # ax.imshow(img)
            # node_names = list(range(len(self.Final_Samples)))
            # end_points = [self.robot.forward_kinematics(config)[-1] for config in self.Final_Samples]
            
            # positions = dict(zip(node_names, end_points))
            # nx.draw(self.graph,positions,node_size=5,edge_color="y",alpha=0.3,ax=ax)
            # List1 = [start,goal]
            # for i in range(2):
            #     # draw robot
            #     self.robot.draw_robot(ax, List1[i], edgecolor="black")
            # plt.axis("on")
            # ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # print("Here2")
            # plt.show()
            # ###############################################

            self.Final_Samples.append(start)
            self.Final_Samples.append(goal)
            self.graph.add_nodes_from(["start", "goal"])
            
            start_goal_tree = spatial.cKDTree([start, goal])
            neighbors = start_goal_tree.query_ball_tree(self.kdtree, 2*self.kdtree_d)
            start_pairs = [["start", neighbor] for neighbor in neighbors[0]]
            goal_pairs = [["goal", neighbor] for neighbor in neighbors[1]]

            print("Start Pairs",start_pairs)
            print("Goal Pairs",goal_pairs)
            
            # Add the edge to graph
            log.I("Adding Start Pairs!")
            self.add_Final_vertices_pairs(start_pairs)
            log.I("Adding Goal Pairs!")
            self.add_Final_vertices_pairs(goal_pairs)
            print("Graph Edges:", self.graph.edges.data())
            # Seach using Dijkstra
            found = False
            length = 0
            try:
                self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                    self.graph, "start", "goal"
                )
                length = (
                    nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
                        self.graph, "start", "goal"
                    )
                )
                found = True
                num_nodes = self.graph.number_of_nodes()
                log.I("The constructed graph has %d of nodes" % num_nodes)
                log.I("The path length is %.2f" % length)
                if length>0:
                    self.solution = [self.Final_Samples[i] for i in self.path[1:-1]]
                    self.solution.insert(0, start)
                    self.solution.append(goal)
                    log.D("Number of Nodes: "+str(len(self.Final_Samples)))
                    fig, ax = plt.subplots()
                    img = 255 * np.dstack((self.map, self.map, self.map))
                    # img = np.flipud(img)
                    ax.imshow(img)
                    node_names = list(range(len(self.Final_Samples)))
                    end_points = [
                        self.robot.forward_kinematics(config)[-1] 
                        for config in self.Final_Samples
                    ]
                    positions = dict(zip(node_names, end_points))
                    positions["start"] = (end_points[-2][1], end_points[-2][0])
                    positions["goal"] = (end_points[-1][1], end_points[-1][0])
                    nx.draw(self.graph,positions,node_size=5,edge_color="y",alpha=0.3,ax=ax)
                    for i in range(len(self.solution)-1):
                        # draw robot
                        for c in self.robot.interpolate(self.solution[i], self.solution[i+1], num=5): 
                            self.robot.draw_robot(ax, c, edgecolor="red")
                    for i in range(len(self.solution)):
                        # draw robot
                        self.robot.draw_robot(ax, self.solution[i], edgecolor="black")
                    plt.axis("on")
                    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                    plt.show()
                    break
                    
                
            except nx.exception.NetworkXNoPath:
                log.I("No path found")
                # log.I("Samples:")
                # for sample in self.samples:
                #     endpoint = self.robot.forward_kinematics(sample)[-1]
                #     print(endpoint)
                # log.I("")
                # log.I("Edges: "+str(self.graph.edges.data()))
            # Remove start and goal node
            self.Final_Samples.pop(-1)
            self.Final_Samples.pop(-1)
        
        
        # self.graph.remove_nodes_from(["start", "goal"])
        # self.graph.remove_edges_from(start_pairs)
        # self.graph.remove_edges_from(goal_pairs)
        Number_of_Nodes = self.graph.number_of_nodes() - 2
        length = round(length,2)
        
            
        # Final solution
        if found:
            self.solution = [self.Final_Samples[i] for i in self.path[1:-1]]
            self.solution.insert(0, start)
            self.solution.append(goal)
        log.D("Solution: "+str(self.solution))
        
                
        return self.solution, length, Number_of_Nodes

    def visualize_sampling_result(self, ax):
        """ Visualization the sampling result."""
        # Get node 2d position by computing robot forward kinematics,
        # assuming the last endpoint of the robot is the position.
        node_names = list(range(len(self.samples)))
        end_points = [
            self.robot.forward_kinematics(config)[-1] 
            for config in self.samples
        ]
        positions = dict(zip(node_names, end_points))

        # Draw constructed graph
        nx.draw(
            self.graph,
            positions,
            node_size=5,
            node_color="g",
            edge_color="y",
            alpha=0.3,
            ax=ax
        )
        
    def return_samples(self):
        return self.samples, self.graph, self.path

