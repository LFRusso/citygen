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
        self.price = slope

        self.undeveloped = True
        self.type = -1
        self.type_color_rgb = (1,1,1)
        self.hatch = None
        
        self.linked_node = None
        self.mesh_distance = None

    # Links the cell to the road network by a given point and the edge it is contained in (by default closest point to the cell)
    def setMeshLink(self, P, d=None):
        self.mesh_distance = d
        if d==None:
            self.mesh_distance = self.distance(P, self.position)

        # Reserve road land use (blocks other type of development in this cell)
        #if d < sqrt(2*self.size**2)/2:
        #    self.setRoad()

        self.linked_node = P
        return

    def distance(self, u, v):
        return np.linalg.norm(np.array(u)-np.array(v))

    def setRoad(self):
        self.type = 0
        self.type_color_rgb = (0,0,0) 
        self.undeveloped = Falses

    # Updates data about current cell based on its type and vicinity
    def update(self, map):
        (i, j) = self.idx
        vicinity = map.cells[max(i-self.K,0) : min(i+self.K+1, map.lines),
                            max(j-self.K,0) : min(j+self.K+1, map.columns)]
        
        if self.type == -1: # Undeveloped
            self.type_color_rgb = (1,1,1)
            self.hatch = None
            self.score = 0

        elif self.type == 0: # Road
            self.type_color_rgb = (0,0,0)
            self.hatch = None
            self.score = 0

        elif self.type == 1: # Residential
            self.type_color_rgb = (0,0,1)
            self.hatch = '...'
            self.score = self.getResidentialScore(vicinity)
 
        elif self.type == 2: # Commercial
            self.type_color_rgb = (1,1,0) 
            self.hatch = '///'
            self.score = self.getCommercialScore(vicinity)

        elif self.type == 3: # Industrial
            self.type_color_rgb = (1,0,0) 
            self.hatch = 'xx'
            self.score = self.getIndustrialScore(vicinity)

        elif self.type == 4: # Recreational
            self.type_color_rgb = (0,1,0) 
            self.hatch = 'OO'
            self.score = 0
        
    def setUndeveloped(self):
        if (self.type==0):
            return
        self.type = -1
        self.undeveloped = True
        self.type_color_rgb = (1,1,1)
        self.score = 0

    # Builds in the current cell while changin its own and vicinity scores 
    def setDeveloped(self, dev_type, map):
        self.type = dev_type
        self.undeveloped = False

        (i, j) = self.idx
        vicinity = map.cells[max(i-self.K,0) : min(i+self.K+1, map.lines),
                            max(j-self.K,0) : min(j+self.K+1, map.columns)]

        self.update(map) # Updating current cell score
        for cell in vicinity.flatten(): # Updating vicinity score
            cell.update(map)