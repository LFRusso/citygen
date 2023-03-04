import numpy as np
from cell import Cell
import matplotlib.pyplot as plt
import time
from scipy.spatial import cKDTree
from tqdm import tqdm
import utm

class Grid:
    def __init__(self, x, y, width, height, cell_size, net=None, searched_nodes=1):
        self.x = x
        self.y = y
        self.width = width
        self.width_multiplier = 1 if width>0 else -1
        self.height = height
        self.height_multiplier = 1 if height>0 else -1
        self.cell_size = cell_size
        self.searched_nodes = searched_nodes

        self.cell_coords = []

        # Number of lines and columns in the grid based on cell size
        self.lines = int(np.ceil(np.abs(height)/cell_size))
        self.columns = int(np.ceil(np.abs(width)/cell_size))

        # Initializing cells
        self.cells = np.empty((self.lines, self.columns), dtype=Cell)
        for i in range(self.lines):
            for j in range(self.columns):
                cell_coords = (self.x + 1*(j*cell_size + cell_size/2), 
                               self.y + 1*(i*cell_size + cell_size/2))

                #cell_coords_latlon = utm.to_latlon(*cell_coords, zone_number, zone_letter)
                #print(cell_coords_latlon, list(heightmap.sample([cell_coords_latlon[::-1]])))
                self.cells[i,j] = Cell(cell_size, cell_coords, (i,j), elevation=None) # 'Position' of the cell is set in its center
        

    def updateNetwork(self, net):
        self.net = net

        start = time.time()
        self._getMeshDistances(self.searched_nodes) # Build edge_sets (cell neighborhoods) based on distance of each sell to each edge
        
        print(f"{self.lines*self.columns} ({self.lines}x{self.columns}) cells built in {time.time() - start}s")


    # Calculates the distance of each cell in relation to the network by finding the closest edge to its center
    def _getMeshDistances(self, n_nodes=1): 
        print("Calculating cell-network distances...")
        flattened_cells = self.cells.flatten()
        cell_coords = [c.position for c in flattened_cells]
        node_coords = [n for n in self.net.nodes]

        # Tries to reduce the searched edges by looking only at those that connect the closest nodes
        node_tree = cKDTree(node_coords) 
        _, idx = node_tree.query(cell_coords, k=n_nodes, workers=-1)
        
        selected_nodes = np.array(node_coords)[idx]
        
        for c, node in tqdm(zip(flattened_cells, selected_nodes), total=len(flattened_cells)):
            c.setMeshLink(node)
            
        return

    def plotGrid(self, links=False):
        for i in range(self.lines):
            for j in range(self.columns):
                plt.gca().add_patch(plt.Rectangle((self.x+self.width_multiplier*j*self.cell_size, 
                                                   self.y+self.height_multiplier*i*self.cell_size), 
                                    self.cell_size, self.cell_size, ec="gray", fc="#e1e1e1", alpha=1, linestyle='-'))
        
        plt.xlim([self.x, self.x+self.width])
        plt.ylim([self.y, self.y+self.height])
        plt.axis("equal")