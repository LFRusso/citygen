import json
import copy

import numpy as np
from perlin_noise import PerlinNoise

with open("config.json") as config_file:
    config = json.load(config_file)

def convertMapMatrix(point):
    return

class Heatmap:
    '''
    def __init__(self, seed, width, height):
        self.noise = PerlinNoise(octaves=3, seed=seed)
        self.width = np.abs(width)
        self.height = np.abs(height)
    '''

    def __init__(self, matrix):
        self.matrix = matrix

    def populationOnRoad(self, road):
        return (self.opulationAt(*road.start) + self.opulationAt(*road.end)) / 2
    
    def opulationAt(self, x, y):
        return self.matrix[int(x), int(y)]
        #return self.noise([x/50, y/50])

class Graph:
    def __init__(self):
        self.edges = {}
        self.nodes = []

class Segment:
    def __init__(self, start, end, time_step=0, highway=False, color="black", severed=False, previous_road=None):
        self.start = start
        self.end = end
        self.time_step = time_step
        self.width = config["HIGHWAY_SEGMENT_WIDTH"] if highway else  config["DEFAULT_SEGMENT_WIDTH"]
        self.severed = severed
        self.color = color
        self.highway = highway
        self.length = config["HIGHWAY_SEGMENT_LENGTH"] if highway else config["DEFAULT_SEGMENT_LENGTH"]
        self.previous_road = previous_road

    def direction(self):
        aux_vec = (self.end[0]-self.start[0], self.end[1]-self.start[1])
        return -1 * np.sign(np.cross((0,1), aux_vec)) * getAngle((0, 1), aux_vec)


class RoadNet:
    def __init__(self, x, y, width, height, price_matrix):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.segments = []
        self.heatmap = Heatmap(price_matrix)
        self.graph = Graph()

        self.highway_count = 0
        self.queue = np.array(self.makeInitialSegment())
        self._sortQueue()
        return

    # Sort the queue based on the time
    def _sortQueue(self):
        times = [s.time_step for s in self.queue]
        self.queue = self.queue[np.argsort(times)][::-1]

    # A single iteration step
    def step(self):
        try:
            self._sortQueue()
            new_road, self.queue = self.queue[-1], self.queue[:-1] # popping
        except:
            #print("ERROR: queue empty")
            return
        
        accepted = self.localConstraints(new_road)
        if (accepted):
            self.segments.append(new_road)
            for next_road in self.globalGoals(new_road):
                next_road.time_step = new_road.time_step + next_road.time_step + 1
                self.queue = np.append(self.queue, next_road)
        return

    def splitSegment(self, segment, point):
        self.segments.remove(segment)
        new_segment_a = Segment(start=segment.start, end=point, time_step=0, 
                                highway=segment.highway, severed=True, previous_road=segment.previous_road)
        new_segment_b = Segment(start=point, end=segment.end, time_step=0, 
                                highway=segment.highway, severed=True, previous_road=segment.previous_road)
        
        self.segments += [new_segment_a, new_segment_b]
        
        self.graph.nodes.append(point)
        self.graph.edges[(segment.start, point)] = {"highway": segment.highway, "color": segment.color}
        self.graph.edges[(point, segment.end)] = {"highway": segment.highway, "color": segment.color}
        del self.graph.edges[(segment.start, segment.end)]
        del(segment)

        return new_segment_a, new_segment_b

    def pointInBounds(self, point):
        if (self.x < point[0] and self.x + self.width > point[0]):
            if (self.y < point[1] and self.y + self.height > point[1]):
                return True
        print(point)
        print(self.x, self.x + self.width)
        print(self.y, self.y + self.height)
        return False

    # Checks if road can be built and do necessary changes according to local constraints
    def localConstraints(self, road):
        def getClosestRoads(road):
            close_roads = []
            start, end = np.array(road.start), np.array(road.end)
            for other in self.segments:
                # Checking distances (can optmize)
                other_start, other_end = np.array(other.start), np.array(other.end)
                min_dist = np.min([np.linalg.norm(start - other_start), np.linalg.norm(end - other_end), 
                                  np.linalg.norm(start - other_end), np.linalg.norm(end - other_start)])
                if (min_dist < config["HIGHWAY_SEGMENT_LENGTH"]):
                    close_roads.append(other)
            return close_roads

        # Checks is road is outside bounds
        if not self.pointInBounds(road.end):
            return False

        # Check if max number of highways reached
        if (road.highway == True):
            if (self.highway_count == config["HIGHWAY_MAX_COUNT"]):
                print("reached")
                return False
            else:
                self.highway_count += 1
                
        if (config["IGNORE_CONFLICTS"]): return True

        closest_roads = getClosestRoads(road)

        # 1. Checking intersects:
        intersecting_roads = []
        for other in closest_roads:
            intersection = intersects(road, other)
            if (intersection != None and (~np.isclose(intersection, road.start)).sum()!=0 and (~np.isclose(intersection, road.end)).sum()!=0):
                intersecting_roads.append(other)

        if (len(intersecting_roads) > 0):
            idx = np.argmin([distance(intersects(road, r), road.start) for r in intersecting_roads])
            other = intersecting_roads[idx]
            intersection = intersects(road, other)
            angle = np.abs(other.direction() - road.direction()) % 180
            angle = min(angle, 180 - angle)
            if angle < config["MINIMUM_INTERSECTION_DEVIATION"]:
                return False

            # Splitting to create intersection
            segment_a, segment_b = self.splitSegment(other, intersection)
            road.end = intersection
            road.severed = True
            road.color = "red"
            self.graph.edges[(road.start, intersection)] = {"highway": road.highway, "color": "red"}
            return True

        # 2. Checking snap to crossing within radius check
        for other in closest_roads:
            if(distance(road.end, other.end) <= config["ROAD_SNAP_DISTANCE"]):
                point = other.end
                road.end = point
                road.severed = True

                road.color = "blue"
                self.graph.edges[(road.start, point)] = {"highway": road.highway, "color": "blue"}
                return True

        # 3. Intersection within radius check
        for other in closest_roads:
            point, distance_segment = distanceToLine(road.end, other)
            if (distance_segment < config["ROAD_SNAP_DISTANCE"]):
                road.end = point
                road.severed = True

                # if intersecting lines are too closely aligned don't continue
                angle = np.abs(other.direction() - road.direction()) % 180
                angle = min(angle, 180 - angle)
                if angle < config["MINIMUM_INTERSECTION_DEVIATION"]:
                    return False
                segment_a, segment_b = self.splitSegment(other, point)
                road.color = "green"
                self.graph.edges[(road.start, point)] = {"highway": road.highway, "color": "green"}
                return True

        # Adding new road to the graph
        if road.start not in self.graph.nodes:
            self.graph.nodes.append(road.start)
        if road.end not in self.graph.nodes:
            self.graph.nodes.append(road.end)
        self.graph.edges[(road.start, road.end)] = {"highway": road.highway, "color": "black"}
        return True

    # Generate next roads according to the global goals from a build road
    def globalGoals(self, road):
        # Road that continues from current, keeping its properties
        def continueRoad(previous_road, direction):
            return segmentFromDirection(previous_road.end, previous_road.direction() + direction, 
                        highway=previous_road.highway, severed=previous_road.severed, length=previous_road.length)
        # Road that branches from previous, having default properties for normal segment
        def branchRoad(previous_road, direction):
            return segmentFromDirection(previous_road.end, previous_road.direction() + direction, 
                        length=config["DEFAULT_SEGMENT_LENGTH"], time_step = config["NORMAL_BRANCH_TIME_DELAY_FROM_HIGHWAY"] if previous_road.highway else 0)

        branches = []
        if not road.severed:
            continue_straight = continueRoad(road, 0)
            straight_pop = self.heatmap.populationOnRoad(continue_straight)

            # Extending the highway
            if road.highway:
                max_pop = straight_pop
                best_segment = continue_straight
                for i in range(config["HIGHWAY_POPULATION_SAMPLE_SIZE"]):
                    current_segment = continueRoad(road, randomStraightAngle())
                    current_pop = self.heatmap.populationOnRoad(current_segment)
                    if (current_pop > max_pop):
                        max_pop = current_pop
                        best_segment = current_segment
                branches.append(best_segment)

                if (max_pop > config["HIGHWAY_BRANCH_POPULATION_THRESHOLD"]):
                    if (np.random.random() < config["HIGHWAY_BRANCH_PROBABILITY"]):
                        branches.append(continueRoad(road, -90 + randomBranchAngle()))
                    if (np.random.random() < config["HIGHWAY_BRANCH_PROBABILITY"]):
                        branches.append(continueRoad(road, 90 + randomBranchAngle()))
            elif straight_pop > config["NORMAL_BRANCH_POPULATION_THRESHOLD"]: #or True: # TO DO: check heatmapp
                branches.append(continue_straight)
            
            # Branching to normal streets
            if not config["ONLY_HIGHWAYS"]:
                if (straight_pop > config["NORMAL_BRANCH_POPULATION_THRESHOLD"]): #or True: # TO DO: check heatmapp
                    if (np.random.random() < config["DEFAULT_BRANCH_PROBABILITY"]):
                        branches.append(branchRoad(road, -90 + randomBranchAngle()))
                        branches.append(branchRoad(road, 90 + randomBranchAngle()))

            
        # Setup links between each current branch and each existing branch stemming from the previous segment   
        for branch in branches:
            branch.previous_road = road
    
        return branches

    # Builds the first one or two network segments
    def makeInitialSegment(self):
        # Starting in the center
        x_center = (2*self.x + self.width) / 2
        y_center = (2*self.y + self.height) / 2

        root = Segment((x_center, y_center), (x_center+config["HIGHWAY_SEGMENT_LENGTH"], y_center), 0, 
                        highway = not config["START_WITH_NORMAL_STREETS"])
        if not config["TWO_SEGMENTS_INITIALLY"]:
            return [root]
        
        root_opposite = copy.deepcopy(root)
        root_opposite.end = (root.start[0] - config["HIGHWAY_SEGMENT_LENGTH"], root.end[1])
        return [root, root_opposite]



def distance(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Checks if two segments intersect. Return intersection point if it exists
def intersects(segment_a, segment_b):
    line1 = (segment_a.start, segment_a.end)
    line2 = (segment_b.start, segment_b.end)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # check if point inside segments
    if (np.isclose(distance(line1[0], line1[1]), distance((x, y), line1[0]) + distance((x, y), line1[1])) and
        np.isclose(distance(line2[0], line2[1]), distance((x, y), line2[0]) + distance((x, y), line2[1]))):
        return (x, y)
    else:
        return None


def distanceToLine(point, segment):
    x, y = point
    p1 = segment.start
    p2 = segment.end

    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy       
    a = ((x-x1)*dx + (y-y1)*dy) / det

    # Making sure points belong to the segment
    a = min(1, max(0, a))
    Px, Py = x1+a*dx, y1+a*dy
    d = distance((Px, Py), (x, y))

    return (Px, Py), d

def segmentFromDirection(start, direction=90, time_step=0, highway=False, color="black", 
                         severed=False, length=config["DEFAULT_SEGMENT_LENGTH"]):
    x = start[0] + length * np.sin((direction * np.pi) / 180)
    y = start[1] + length * np.cos((direction * np.pi) / 180)

    return Segment(start, (x, y), time_step=time_step, highway=highway, color=color, severed=severed)

def getAngle(u, v):
    dot = np.dot(u, v)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    cos_x = dot/(norm_u*norm_v)
    return np.rad2deg(np.arccos(cos_x))

def randomStraightAngle():
    return np.random.uniform(-15, 15)

def randomBranchAngle():
    return np.random.uniform(-0, 0)
    #return np.random.uniform(-3, 3)