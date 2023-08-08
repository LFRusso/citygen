from numpy import sqrt, exp
import numpy as np

class Cell:
    def __init__(self, size, pos, idx, elevation, slope, K=5):
        self.x = pos[0]
        self.y = pos[1]
        self.position = pos
        self.idx = idx
        self.size = size
        self.elevation = elevation
        self.K = K
        self.slope = slope
        self.price = np.exp(-slope)*3000
        #self.price = np.exp(-elevation)*3000

        self.empty = True
        self.type = -1
        self.type_color_rgb = (0,0,0,0)
        self.hatch = None
        
        self.linked_node = None
        self.mesh_distance = None

    # TO DO
    def updatePrice(self, price):
        self.price = price
        return

    # Links the cell to the road network by a given point and the edge it is contained in (by default closest point to the cell)
    def setMeshLink(self, P, d=None):
        self.mesh_distance = d
        if d==None:
            self.mesh_distance = self.distance(P, self.position)

        self.price = 7000 / (1 + self.mesh_distance/1.5)
        #self.price = 7000 * np.exp(-self.mesh_distance)
        #print(self.price, self.mesh_distance)
        self.linked_node = P
        return

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))

    # Updates data about current cell
    def update(self):
        if self.type == -1: # Empty
            self.type_color_rgb = (200,200,200, 0)
            self.hatch = None

        elif self.type == 1: # Residential
            self.type_color_rgb = (0,0,255, 255)
            self.hatch = '...'
 
        elif self.type == 2: # Commercial
            self.type_color_rgb = (255,255,0,255) 
            self.hatch = '///'

        elif self.type == 3: # Industrial
            self.type_color_rgb = (255,0,0, 255) 
            self.hatch = 'xx'

        elif self.type == 4: # Recreational
            self.type_color_rgb = (0,255,0, 255) 
            self.hatch = 'OO'
        
    def setEmpty(self):
        self.type = -1
        self.empty = True
        self.update()

    # Builds in the current cell
    def setDeveloped(self, dev_type):
        self.type = dev_type
        self.empty = False
        self.update()