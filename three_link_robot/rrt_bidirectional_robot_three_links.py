#!/usr/bin/env python3
#
#   prmrobot.py
#
#   Use PRM to find a path for the planar three link robot.
#   This does not allow wrapping joints cannot move past +/- 180 degrees
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math          import pi, sin, cos, sqrt, ceil
from scipy.spatial import KDTree
from shapely.geometry   import Point, LineString, MultiLineString, Polygon
from shapely.prepared   import prep

import vandercorput


######################################################################
#
#   Parameters
DSTEP = 10.0

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000 * 2
NMAX = 1500 * 2

######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
(xmin, xmax) = (-3.5, 3.5)
(ymin, ymax) = (-1.5, 3.5)

(xL, xR)     = (-0.5, 0.5)
(yB, yL, yR) = (-0.5, 1.5, 1.0)

xlabels = (-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5)
ylabels = (-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5)

# Collect all the walls
walls = MultiLineString([[[xmin, yB], [xmax, yB]],
                         [[xmin, yL], [xL, yL], [xL, ymax]],
                         [[xR, yR], [xR, ymax]]])

# Define the start/goal states (joint values)
(startq1, startq2, startq3) = (0, 0, 0)
(goalq1,  goalq2,  goalq3)  = (pi/2, 0, 0)

# FIXME: Define any distances you need to use, in physical or joint spaces?
Dx = 0.1 #dmin = 0.1m in the lecture video
Dq = Dx/3.0 #(dmin/length of arm = dmin/3m)

Dqdraw = 0.5    # Joint distance to space robots while drawing path


######################################################################
#
#   Utilities: Angle Wrapping and Visualization
#

# Angle Wrap Utility.  Return the angle wrapped into +/- 1/2 of full range.
def wrap(angle, fullrange):
    return angle - fullrange * round(angle/fullrange)

# Visualization Class.
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
        plt.gca().set_xticks(xlabels)
        plt.gca().set_yticks(ylabels)
        plt.gca().set_aspect('equal')

        # Show the walls.
        for l in walls.geoms:
            plt.plot(*l.xy, color='k', linewidth=2)

        # Place joint 0 only once!
        plt.gca().add_artist(plt.Circle((0,0),color='k',radius=0.05))

        # Show.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawTip(self, node, *args, **kwargs):
        plt.arrow(node.xB, node.yB, 0.9*(node.xC-node.xB), 0.9*(node.yC-node.yB),
                  head_width=0.1, head_length=0.1, *args, **kwargs)

    def drawRobot(self, node, *args, **kwargs):
        plt.plot((0, node.xA, node.xB), (0, node.yA, node.yB), *args, **kwargs)
        self.drawTip(node, *args, **kwargs)
        kwargs['radius']=0.05
        plt.gca().add_artist(plt.Circle((node.xA,node.yA),*args,**kwargs))
        plt.gca().add_artist(plt.Circle((node.xB,node.yB),*args,**kwargs))

    def drawNode(self, node, *args, **kwargs):
        self.drawTip(node, *args, **kwargs)

    def drawEdge(self, n1, n2, *args, **kwargs):
        plt.plot(((n1.xB + n1.xC)/2, (n2.xB + n2.xC)/2),
                 ((n1.yB + n1.yC)/2, (n2.yB + n2.yC)/2), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            n = ceil(path[i].distance(path[i+1]) / Dqdraw)
            for j in range(n):
                node = path[i].intermediate(path[i+1], j/n)
                self.drawRobot(node, *args, **kwargs)
                plt.pause(0.1)
        self.drawRobot(path[-1], *args, **kwargs)


######################################################################
#
#   Node Definition
#
class Node():
    def __init__(self, q1, q2, q3):
        self.parent = None

        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        # Pre-compute the link positions.
        (self.xA, self.yA) = (          cos(q1)      ,           sin(q1)      )
        (self.xB, self.yB) = (self.xA + cos(q1+q2)   , self.yA + sin(q1+q2)   )
        (self.xC, self.yC) = (self.xB + cos(q1+q2+q3), self.yB + sin(q1+q2+q3))
        self.links = LineString([[0,0], [self.xA,self.yA],
                                 [self.xB,self.yB], [self.xC, self.yC]])

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Joints %6.1fdeg,%6.1fdeg,%6.1fdeg>" %
                (self.q1 * 180/pi, self.q2 * 180/pi, self.q3 * 180/pi))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    # Note, how to you move from 350deg to 370deg?  Is this a +20deg
    # movement?  Or is 370deg=10deg and this is a -240deg movement?
    def intermediate(self, other, alpha):
        #FIXME: Please implement
        #case a)
        return Node(self.q1 + alpha *  (other.q1 - self.q1),
                    self.q2 + alpha *  (other.q2 - self.q2),
                    self.q3 + alpha *  (other.q3 - self.q3))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        #FIXME: Please implement
        #case a)
        return (self.q1, self.q2, self.q3)

    # Compute the relative distance to another node.  See above.
    def distance(self, other):
        #FIXME: Please implement
        #case a)
        return sqrt((self.q1 - other.q1)**2 + (self.q2 - other.q2)**2 +
                    (self.q3 - other.q3)**2)


    ###############
    # A* functions:
    # Actual and Estimated costs.
    def costToConnect(self, other):
        return self.distance(other)

    def costToGoEst(self, other):
        return self.distance(other)

    ################
    # PRM functions:
    # Check whether in free space.
    def inFreespace(self):
        #FIXME: return True if you are know the arm is not hitting any wall.
        return walls.disjoint(self.links)

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        #FIXME: return True if you can move without collision.
        #case a)
        if walls.distance(self.links) < Dx or walls.distance(other.links) < Dx:
            return False

        for delta in vandercorput.sequence(Dq / self.distance(other)):
            intermediate_node = self.intermediate(other, delta)
            if not intermediate_node.inFreespace():
                return False
            if walls.distance(intermediate_node.links) < Dx:
                return False

        return True

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
            q1_coord = random.uniform(-pi, pi)
            q2_coord = random.uniform(-pi, pi)
            q3_coord = random.uniform(-pi, pi)
            targetnode = Node(q1_coord, q2_coord, q3_coord)

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree_a])
        index     = np.argmin(distances)
        nearnode  = tree_a[index] #nearest node to target
        d         = distances[index] #distance between nearest node and target

        # Determine the next node.
        if d <= DSTEP:
            nextnode = targetnode
        else:
            # get intermediate node
            nextnode = nearnode.intermediate(targetnode, DSTEP/d)

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

# Post Process the Path
def PostProcess(path):
    #FIXME: remove unnecessary nodes from the path, if the predecessor and
    #       successor can connect directly.  I.e. minimize the steps.
    ref_node = path[0]
    skipped_nodes = []
    if len(path) > 2:
        for i in range(1, len(path)-1):
            if ref_node.connectsTo(path[i+1]):
                skipped_nodes.append(path[i])
            else:
                ref_node = path[i]
        for node in skipped_nodes:
            path.remove(node)


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
    startnode = Node(startq1, startq2, startq3)
    goalnode  = Node(goalq1,  goalq2,  goalq3)

    # Show the start/goal nodes.
    visual.drawRobot(startnode, color='orange', linewidth=2)
    visual.drawRobot(goalnode,  color='purple', linewidth=2)
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


    # Post Process the path.
    PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")

    # Report the path.
    print(path[0])
    for i in range(1,len(path)):
        path[i] = path[i-1].intermediate(path[i], 1.0)
        print(path[i])
    

if __name__== "__main__":
    main()
