import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


class Planner:
    """A planner class that plans a path from start to goal
    in a certain map for a certain robot using a certain method.
    """

    def __init__(self, method, map_2d, robot):
        """Initialize with the method, robot, map, and obstacles."""
        self.method = method
        self.map = map_2d
        self.robot = robot
        self.pos = {}
        self.position_array = []

        # initialize the method with the map and robot
        # Change set_map and set_robot functions
        self.method.set_map(self.map)
        self.method.set_robot(self.robot)

        # result container
        self.start = None
        self.goal = None
        self.solution = None

    def plan(self, start, goal):
        """Plan a path from start to goal."""
        self.start = start
        self.goal = goal
        self.solution, path_Length, Number_of_Nodes = self.method.test_plan(self.start, self.goal)
        return self.solution, path_Length, Number_of_Nodes

    def visualize(self, make_video=False):
        """Visualize the envrionment and solution."""
        # Draw the map
        ax, fig = self.draw_map()

        # Define the meta data for the movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='An animation of the robot planning')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        # self.method.visualize_sampling_result(ax)
        
        # Draw the robot with solution configurations
        if self.solution != []:
               if make_video:
                   writer.setup(fig, "D:\WPI\Sem1\Motion_Planning\Final_Project\path.mp4", 100)

               # draw path
               for i in range(len(self.solution)):
                  if i == len(self.solution) - 1:
                       continue
                  p1 = self.robot.forward_kinematics(self.solution[i])[-1]
                  p2 = self.robot.forward_kinematics(self.solution[i + 1])[-1]
                  ax.plot(
                      [p1[0], p2[0]],
                      [p1[1], p2[1]],
                      color="gray",
                      linewidth=1,
                  )
                  if make_video:
                      writer.grab_frame()

               for i in range(len(self.solution)-1):
                   # draw robot
                   for c in self.robot.interpolate(self.solution[i], self.solution[i+1], num=10): 
                       self.robot.draw_robot(ax, c , edgecolor="blue")
                       if make_video:
                           writer.grab_frame()

        # Draw the start and goal configuration if provided
        if self.start is not None and self.goal is not None:
            self.robot.draw_robot(ax, self.start, edgecolor="red")
            self.robot.draw_robot(ax, self.goal, edgecolor="green")

    def draw_map(self):
        """Visualization of the result"""
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map, self.map, self.map))
        # img = np.flipud(img)
        ax.imshow(img)

        samples,graph,path = self.method.return_samples()
        node_names = list(range(len(samples)))
        end_points = [
            self.robot.forward_kinematics(config)[-1] 
            for config in samples
        ]
        positions = nx.spring_layout(graph)
        positions = dict(zip(node_names, end_points))

        positions["start"] = (end_points[-2][1], end_points[-2][0])
        positions["goal"] = (end_points[-1][1], end_points[-1][0])
        # Draw constructed graph
        nx.draw(
            graph,
            positions,
            node_size=5,
            node_color="g",
            edge_color="y",
            alpha=0.3,
            ax=ax
        )
        if self.solution:
            for i in range(len(self.solution)-1):
                # draw robot
                for c in self.robot.interpolate(self.solution[i], self.solution[i+1], num=10): 
                    self.robot.draw_robot(ax, c, edgecolor="red")
            for i in range(len(self.solution)):
                # draw robot
                self.robot.draw_robot(ax, self.solution[i], edgecolor="black")
        # show image
        plt.axis("on")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # ax.invert_yaxis()
        plt.show()
        return ax, fig