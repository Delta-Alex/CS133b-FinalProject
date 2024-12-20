#!/usr/bin/env python3
#
#   rrttriangles.py
#
#   Use RRT to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import pandas as pd  

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
RADIUS = DSTEP * 3
MIN_DSTEP = 0.25
K = 0.50 #factor to control step size

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000 * 4
NMAX = 1500 * 5


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
    Polygon([[ 2, 6], [ 3, 2], [ 4, 6], [ 2, 6]]),
    Polygon([[ 6, 5], [ 7, 7], [ 8, 5], [ 6, 5]]),
    Polygon([[ 6, 9], [ 8, 9], [ 6, 7], [ 6, 9]]),
    Polygon([[10, 3], [11, 6], [12, 3], [10, 3]])]))

# Define the start/goal states (x, y, theta)
(xstart, ystart) = (13, 5)
(xgoal,  ygoal)  = ( 1, 5)


RUN_TEST = False
DRAW = True
N = 0
rrt_types = ["rrt_reg", "rrt_march", "rrt_var", "rrt_star", "rrt_bi"]
if len(sys.argv) > 1:
    grid = sys.argv[1]
    N = int(sys.argv[2])
    rrt_type = sys.argv[3].lower()
    if rrt_type not in rrt_types:
        raise Exception("Invalid RRT Type")
    if grid == "grid1":
        # narrow grid
        (xmin, xmax) = (0, 14) 
        (ymin, ymax) = (0, 10)

        # Collect all the triangle and prepare (for faster checking).
        multi = MultiPolygon([
            Polygon([[4, 1], [10, 1], [10, 9], [4, 9], [4, 1]]),
            Polygon([[10, 2], [12, 2], [10, 5], [10, 2]]),
            Polygon([[10, 8], [12, 8], [10, 5], [10, 8]])])

        
        triangles = prep(multi)
        obstacles = list(multi.geoms)

        # Define the start/goal states (x, y, theta)
        (xstart, ystart) = ( 1, 5)
        (xgoal,  ygoal)  = (13, 5)
        RUN_TEST = True
        DRAW = False

    elif grid == "grid2":
        #face grid (few obtacles)
        (xmin, xmax) = (0, 14) #Grid size for narrow is 0, 10
        (ymin, ymax) = (0, 14)

        # Collect all the triangle and prepare (for faster checking).
        multi = MultiPolygon([
            Polygon([[4, 3], [10, 3], [10, 5], [4, 5], [4, 3]]),
            Polygon([[4, 11], [6, 11], [5, 9], [4, 11]]),
            Polygon([[8, 11], [10, 11], [9, 9], [8, 11]])])

        triangles = prep(multi)
        obstacles = list(multi.geoms)

        # Define the start/goal states (x, y, theta)
        (xstart, ystart) = ( 1, 12)
        (xgoal,  ygoal)  = (13, 4)
        RUN_TEST = True
        DRAW = False


    elif grid == "grid3":
        DSTEP = 0.50
        #maze like grid
        (xmin, xmax) = (0, 14) #Grid size for narrow is 0, 10
        (ymin, ymax) = (0, 14)

        # Collect all the triangle and prepare (for faster checking).
        multi = MultiPolygon([
            Polygon([[4, 14], [7, 14], [7, 13], [4, 13], [4, 14]]),
            Polygon([[2, 11], [4, 11], [4, 9], [2, 9], [2, 11]]),
            Polygon([[6, 12], [10, 12], [10, 10], [6, 10], [6, 12]]),
            Polygon([[10, 12], [12, 12], [12, 2], [10, 2], [10, 12]]),
            Polygon([[7, 8], [10, 8], [10, 6], [7, 6], [7, 8]]),
            Polygon([[0, 7], [2, 7], [2, 5], [0, 5], [0, 7]]),
            Polygon([[3, 5], [5, 5], [5, 3], [3, 3], [3, 5]]),
            Polygon([[0, 2], [2, 2], [2, 0], [0, 0], [0, 2]]),
            Polygon([[4, 2], [12, 2], [12, 0], [4, 0], [4, 2]])])

        triangles = prep(multi)
        obstacles = list(multi.geoms)

        # Define the start/goal states (x, y, theta)
        (xstart, ystart) = ( 1, 13)
        (xgoal,  ygoal)  = (13, 1)
        RUN_TEST = True
        DRAW = False

    else:
        raise Exception("Invalid grid")
        
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
        return plt.plot((head.x, tail.x),
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
        self.edge_to_parent = None #store plt line from Parent to Node

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

        self.cost = 0

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

    def CostToConnect(self,other):
        return self.distance(other)

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

    #gets the min distance the point is from an obstacle
    def min_obs_xdist(self):
        min_dist = sys.maxsize
        point = Point(self.x,self.y)
        for obs in obstacles:
            dist = obs.exterior.distance(point)
            if dist < min_dist:
                min_dist = dist

        return max(MIN_DSTEP, K * min_dist)


######################################################################
#
#   RRT Functions
#
def rrt(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if DRAW:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    # Loop - keep growing the tree.
    steps = 0
    while True:
        #probability that goal is selected as target, 
        #adjusted as necessary for the given problem
        goal_p = 0.05
        val = np.random.choice(a = np.array([0,1]), p = np.array([goal_p, 1-goal_p]))

        if val == 0:
            targetnode = goalnode

        else:
            x_coord = random.uniform(xmin, xmax)
            y_coord = random.uniform(ymin, ymax)
            targetnode = Node(x_coord, y_coord)

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index] #nearest node to target
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
            addtotree(nearnode, nextnode)

            # If within DSTEP, also try connecting to the goal.  If
            # the connection is made, break the loop to stop growing.
            #FIXME:
            dist_togoal = nextnode.distance(goalnode)
            if dist_togoal <= DSTEP and nextnode.connectsTo(goalnode):
                addtotree(nextnode, goalnode)
                break

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            return None, None, None

    # Build the path.
    path = [goalnode]
    i = 0
    max_i = len(tree) + 1
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
        i += 1
        # there is a cycle
        if i > max_i:
            return None, None, None

    # Report and return.
    #print("Finished after %d steps and the tree having %d nodes" %
    #      (steps, len(tree)))
    return path, steps, len(tree)

def rrt_march(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if DRAW:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    # Loop - keep growing the tree.
    steps = 0
    found_goal = False
    while not found_goal:
        #probability that goal is selected as target, 
        #adjusted as necessary for the given problem
        goal_p = 0.05
        val = np.random.choice(a = np.array([0,1]), p = np.array([goal_p, 1-goal_p]))

        if val == 0:
            targetnode = goalnode

        else:
            x_coord = random.uniform(xmin, xmax)
            y_coord = random.uniform(ymin, ymax)
            targetnode = Node(x_coord, y_coord)

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index] #nearest node to target
        d         = distances[index] #distance between nearest node and target

        # Determine the next node.
        if d <= DSTEP:
            nextnode = targetnode

            # Check whether to attach.
            if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
                addtotree(nearnode, nextnode)

                # If within DSTEP, also try connecting to the goal.  If
                # the connection is made, break the loop to stop growing.
                #FIXME:
                dist_togoal = nextnode.distance(goalnode)
                if dist_togoal <= DSTEP and nextnode.connectsTo(goalnode):
                    addtotree(nextnode, goalnode)
                    found_goal = True

            # Check whether we should abort - too many steps or nodes.
            steps += 1
            if (steps >= SMAX) or (len(tree) >= NMAX):
                #print("Aborted after %d steps and the tree having %d nodes" %
                #    (steps, len(tree)))
                return None, None, None
        else:
            # take repeated steps to target node
            # until either the target is reached or
            # there is an obstacle
            while True:
                # directional vector between nearnode and targetnode
                dir_vec = np.array([targetnode.x - nearnode.x, targetnode.y - nearnode.y])
                dir_vec = dir_vec/np.linalg.norm(dir_vec)
                new_x = nearnode.x + DSTEP * dir_vec[0]
                new_y = nearnode.y + DSTEP * dir_vec[1]
                nextnode = Node(new_x, new_y)
                dist_to_target = nextnode.distance(targetnode)

                # Check whether to attach.
                if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
                    addtotree(nearnode, nextnode)

                    # If within DSTEP, also try connecting to the goal.  If
                    # the connection is made, break the loop to stop growing.
                    dist_togoal = nextnode.distance(goalnode)
                    dist_to_target = nextnode.distance(targetnode)
                    if dist_togoal <= DSTEP and nextnode.connectsTo(goalnode):
                        addtotree(nextnode, goalnode)
                        found_goal = True
                        break

                    elif dist_to_target <= DSTEP and nextnode.connectsTo(targetnode):
                        addtotree(nextnode, targetnode)
                        break

                    # Check whether we should abort - too many steps or nodes.
                    steps += 1
                    if (steps >= SMAX) or (len(tree) >= NMAX):
                        #print("Aborted after %d steps and the tree having %d nodes" %
                        #    (steps, len(tree)))
                        return None, None, None
                    nearnode = nextnode
                else: 
                    break


    # Build the path.
    path = [goalnode]
    i = 0
    max_i = len(tree) + 1
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
        i += 1
        # there is a cycle
        if i > max_i:
            return None, None, None

    # Report and return.
    #print("Finished after %d steps and the tree having %d nodes" %
    #      (steps, len(tree)))
    return path, steps, len(tree)


def rrt_var(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if DRAW:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    # Loop - keep growing the tree.
    steps = 0
    while True:
        #probability that goal is selected as target, 
        #adjusted as necessary for the given problem
        goal_p = 0.05
        val = np.random.choice(a = np.array([0,1]), p = np.array([goal_p, 1-goal_p]))

        if val == 0:
            targetnode = goalnode

        else:
            x_coord = random.uniform(xmin, xmax)
            y_coord = random.uniform(ymin, ymax)
            targetnode = Node(x_coord, y_coord)

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index] #nearest node to target
        d         = distances[index] #distance between nearest node and target

        # Determine the next node.
        DSTEP = nearnode.min_obs_xdist()
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
            addtotree(nearnode, nextnode)

            # If within DSTEP, also try connecting to the goal.  If
            # the connection is made, break the loop to stop growing.
            dist_togoal = nextnode.distance(goalnode)
            if dist_togoal <= DSTEP and nextnode.connectsTo(goalnode):
                addtotree(nextnode, goalnode)
                break

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            #print("Aborted after %d steps and the tree having %d nodes" %
            #      (steps, len(tree)))
            return None, None, None

    # Build the path.
    path = [goalnode]
    i = 0
    max_i = len(tree) + 1
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
        i += 1
        # there is a cycle
        if i > max_i:
            return None, None, None

    # Report and return.
    #print("Finished after %d steps and the tree having %d nodes" %
    #      (steps, len(tree)))
    return path, steps, len(tree)


def rrt_star(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]
    def addtotree(oldnode, newnode, cost):
        newnode.parent = oldnode
        newnode.cost = cost
        tree.append(newnode)
        if DRAW:
            edge = visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            newnode.edge_to_parent = edge
            visual.show()

    def updatetree(oldnode, newnode, cost):
        newnode.parent = oldnode
        newnode.cost = cost
        if DRAW:
            edge = visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            newnode.edge_to_parent = edge
            visual.show()
    
    def get_neighbors(newnode, dist):
        neighbors = []
        for node in tree:
            distance = newnode.distance(node)
            if distance <= dist and dist > 0:
                neighbors.append(node)
        return neighbors


    # Loop - keep growing the tree.
    steps = 0
    while True:
        #probability that goal is selected as target, 
        #adjusted as necessary for the given problem
        goal_p = 0.05
        val = np.random.choice(a = np.array([0,1]), p = np.array([goal_p, 1-goal_p]))

        if val == 0:
            targetnode = goalnode

        else:
            x_coord = random.uniform(xmin, xmax)
            y_coord = random.uniform(ymin, ymax)
            targetnode = Node(x_coord, y_coord)

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index] #nearest node to target
        d         = distances[index] #distance between nearest node and target

        # Determine the next node.
        #FIXME:
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
            # get the neighbors with a certain distance of nextnode
            next_neighbors = get_neighbors(nextnode, RADIUS) 
            #tree.append(nextnode)
            min_node = nearnode
            min_cost = nearnode.cost + nearnode.CostToConnect(nextnode)

            # check if there is a short path to nextnode
            for neighbor in next_neighbors:
                if neighbor.connectsTo(nextnode):
                    newcost = neighbor.cost + neighbor.CostToConnect(nextnode)
                    if newcost < min_cost:
                        min_node = neighbor
                        min_cost = newcost
                   
            addtotree(min_node, nextnode, min_cost)

            for neighbor in next_neighbors:
                if nextnode.connectsTo(neighbor):
                    newcost = nextnode.cost + nextnode.CostToConnect(neighbor)
                    if newcost < neighbor.cost:
                        if DRAW:
                            # remove edge from parent to neighbor
                            line = neighbor.edge_to_parent
                            edge = line.pop(0)
                            edge.remove()

                        # add edge with less cost to reach neighbor
                        updatetree(nextnode, neighbor, newcost)
            
            dist_togoal = nextnode.distance(goalnode)
            if dist_togoal <= DSTEP and nextnode.connectsTo(goalnode):
                cost = nextnode.cost + dist_togoal
                addtotree(nextnode, goalnode, cost)
                break
        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            #print("Aborted after %d steps and the tree having %d nodes" %
            #      (steps, len(tree)))
            return None, None, None

    # Build the path.
    path = [goalnode]
    i = 0
    max_i = len(tree) + 1
    while path[0].parent is not None:
        path.insert(0, path[0].parent)
        i += 1
        # there is a cycle
        if i > max_i:
            return None, None, None

    # Report and return.
    #print("Finished after %d steps and the tree having %d nodes" %
    #      (steps, len(tree)))
    return path, steps, len(tree)


def rrt_bi(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    goalnode.parent = None
    tree_a = [startnode]
    tree_b = [goalnode]

    def addtotree(tree, oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if DRAW:
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
                        path.insert(0, nextnode)
                        while path[0].parent is not None:
                            path.insert(0, path[0].parent)

                        path.append(node)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                    else:
                        path.insert(0, node)
                        while path[0].parent is not None:
                            path.insert(0, path[0].parent)
                        
                        path.append(nextnode)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                    # Report and return path
                    num_nodes = len(tree_a) + len(tree_b)
                    #print("Finished after %d steps and the tree having %d nodes" %
                    #    (steps, num_nodes))

                    return path, steps, num_nodes

        
        tree_a, tree_b = tree_b, tree_a

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        num_nodes = len(tree_a) + len(tree_b)
        if (steps >= SMAX) or (num_nodes >= NMAX):
            #print("Aborted after %d steps and the tree having %d nodes" %
            #      (steps, num_nodes))
            break
  
    return None, None, None


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
    path, steps, num_nodes = rrt(startnode, goalnode, visual)

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


"""
Runs test output csv file with
number of nodes, 
"""
def run_test():
    # create three lists for employee name, id, and salary
    num_nodes_lst = [] 
    steps_lst = [] 
    dist_lst = []
    dt_lst = []
    succesful_runs = 0
    while succesful_runs < N:
        # Create the start/goal nodes.
        startnode = Node(xstart, ystart)
        goalnode  = Node(xgoal,  ygoal)

        if DRAW:
            # Create the figure.
            visual = Visualization()

            # Show the start/goal nodes.
            visual.drawNode(startnode, color='orange', marker='o')
            visual.drawNode(goalnode,  color='purple', marker='o')
            visual.show("Showing basic world")

        else:
            visual = None


        # Run the RRT planner.
        dt = 0.0
        if rrt_type == "rrt_reg":
            start = time.time()
            path, steps, num_nodes = rrt(startnode, goalnode, visual)
            dt = time.time() - start
        elif rrt_type == "rrt_march":
            start = time.time()
            path, steps, num_nodes = rrt_march(startnode, goalnode, visual)
            dt = time.time() - start
        elif rrt_type == "rrt_var":
            start = time.time()
            path, steps, num_nodes = rrt_var(startnode, goalnode, visual)
            dt = time.time() - start
        elif rrt_type ==  "rrt_star":
            start = time.time()
            path, steps, num_nodes = rrt_star(startnode, goalnode, visual)
            dt = time.time() - start
        elif rrt_type == "rrt_bi":
            start = time.time()
            path, steps, num_nodes = rrt_bi(startnode, goalnode, visual)
            dt = time.time() - start
        else:
            continue

        # If unable to connect, just note before closing.
        if not path:
            if DRAW:
                visual.show("UNABLE TO FIND A PATH")
            continue

        else:
            succesful_runs += 1
            dist = 0
            for i in range(len(path) - 1):
                dist += path[i].distance(path[i+1])
            print("Run %d in %f secs with %d steps, %d nodes, and dist %f" % (succesful_runs, dt, steps, num_nodes, dist))
            num_nodes_lst.append(num_nodes)
            steps_lst.append(steps)
            dist_lst.append(dist)
            dt_lst.append(dt)
       
        if DRAW:
            # Show the path.
            visual.drawPath(path, color='r', linewidth=2)
            visual.show("Showing the raw path")

            # Post process the path.
            PostProcess(path)

            # Show the post-processed path.
            visual.drawPath(path, color='b', linewidth=2)
            visual.show("Showing the post-processed path")
    

    # create a dictionary with the three lists
    dict = {'Nodes': num_nodes_lst, 'Steps': steps_lst, 'Dist': dist_lst, 'Time': dt_lst}  
        
    # create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(dict) 

    # write the DataFrame to a CSV file
    filename = "{}_{}_2d_point.csv".format(rrt_type, grid)
    df.to_csv(filename, index = False)


if __name__== "__main__":
    if RUN_TEST:
        #DRAW = TRUE
        run_test()
    else:
        main()
