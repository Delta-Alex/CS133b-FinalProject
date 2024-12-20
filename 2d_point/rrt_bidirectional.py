#!/usr/bin/env python3
#
#   rrttriangles.py
#
#   Use RRT to find a path around triangular obstacles.
#
from http.client import NETWORK_AUTHENTICATION_REQUIRED
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math               import pi, sin, cos, atan2, sqrt, ceil
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep


######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#
DSTEP = 0.25

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000
NMAX = 1500


######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (0, 14)
(ymin, ymax) = (0, 10)

# Collect all the triangle and prepare (for faster checking).
triangles = prep(MultiPolygon([
    Polygon([[5, 3], [5, 7], [9, 7], [9, 3], [5,3]])]))

# Define the start/goal states (x, y, theta)
(xstart, ystart) = (13, 9)
(xgoal,  ygoal)  = ( 1, 1)


######################################################################
#
#   Utilities: Visualization
#
# Visualization Class
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for poly in triangles.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        # Show.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)


######################################################################
#
#   Node Definition
#
class Node:
    def __init__(self, x, y):
        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    ################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        return triangles.disjoint(Point(self.coordinates()))

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([self.coordinates(), other.coordinates()])
        return triangles.disjoint(line)


######################################################################
#
#   Bi directional RRT Function
#
def rrt(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    goalnode.parent = None
    tree_a = [startnode]
    tree_b = [goalnode]

    def addtotree(tree, oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        visual.show()

    # Loop - keep growing the tree.
    steps = 0
    while True:
        #probability that goal is selected as target, 
        #adjusted as necessary for the given problem
        # the goal will change depending if we are extending 
        # the tree with root = startnode or 
        # the tree with root = goalnode
        goal_p = 0.05
        val = np.random.choice(a = np.array([0,1]), p = np.array([goal_p, 1-goal_p]))

        if val == 0:
            targetnode = goalnode if steps % 2 == 0 else startnode

        else:
            x_coord = random.uniform(xmin, xmax)
            y_coord = random.uniform(ymin, ymax)
            targetnode = Node(x_coord, y_coord)

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree_a])
        index     = np.argmin(distances)
        nearnode  = tree_a[index] #nearest node to target
        d         = distances[index] #distance between nearest node and target

        # Determine the next node.
        if d <= DSTEP:
            nextnode = targetnode
        else:
            # directional vector between nearnode and targetnode
            dir_vec = np.array([targetnode.x - nearnode.x, targetnode.y - nearnode.y])
            dir_vec = dir_vec/np.linalg.norm(dir_vec)
            new_x = nearnode.x + DSTEP * dir_vec[0]
            new_y = nearnode.y + DSTEP * dir_vec[1]
            nextnode = Node(new_x, new_y)

        # Check whether to attach.
        if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
            addtotree(tree_a, nearnode, nextnode)
            #print("goal parent: {}".format(goalnode.parent))

            for node in tree_b:
                # check if node can connect to nextnode
                dist = node.distance(nextnode)
                if dist <= 1.0 and node.connectsTo(nextnode):
                    #build the path
                    path = []

                    if steps % 2 == 0:
                        #print("First Case")
                        # Get path from start to nextnode
                        path.insert(0, nextnode)
                        while path[0].parent is not None:
                            path.insert(0, path[0].parent)

                        #print("Done with first segment")

                        # Now append the path from node to goal
                        path.append(node)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                        #print("Done with second segment")

                    else:
                        #print("Second Case")
                        # Get path from start to node
                        path.insert(0, node)
                        while path[0].parent is not None:
                            path.insert(0, path[0].parent)
                        
                        #print("Done with first segment")

                        # Now append the path from nextnode to goal
                        path.append(nextnode)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                        #print("Done with second segment")


                    # Report and return path
                    num_nodes = len(tree_a) + len(tree_b)
                    print("Finished after %d steps and the tree having %d nodes" %
                        (steps, num_nodes))

                    return path

        
        tree_a, tree_b = tree_b, tree_a

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        num_nodes = len(tree_a) + len(tree_b)
        if (steps >= SMAX) or (num_nodes >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, num_nodes))
            break
  
    return None


# Post process the path.
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1


######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    # Create the figure.
    visual = Visualization()

    # Create the start/goal nodes.
    startnode = Node(xstart, ystart)
    goalnode  = Node(xgoal,  ygoal)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    path = rrt(startnode, goalnode, visual)

    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path")


    # Post process the path.
    PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()
