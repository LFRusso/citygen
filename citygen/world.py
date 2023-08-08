import utm
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import elevation as elev
import os
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
#from osgeo import gdal
import pyproj
import time
from scipy.spatial import cKDTree
from tqdm import tqdm

from plot import plotRoad
from cell import Cell

CWD = os.getcwd()
PROJECTION = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS('EPSG:3857'), always_xy=True).transform


def reproject_raster(in_path, out_path):
    crs = "EPSG:3857"
    # reproject raster to project crs
    with rio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear)
    
    raster = rio.open(out_path)
    return raster

def project_point(point):
    return PROJECTION(*point)

class World:
    def __init__(self, place, elevation_file, cell_size=50):
        self.place = place
        self.land = ox.geocode_to_gdf(place) # Obtaining geographical data from omnx
        #self.bounds_latlon = self.land.total_bounds 
        #self.bounds_latlon = [-48.139751, -23.3830599, -47.949, -23.2224164]
        #self.bounds_latlon = [-74.047207  40.679654 -73.906769  40.882012]
        self.bounds_latlon = [-49.05, -22.3, -49.0, -22.25]
        print(self.land.total_bounds)
        self.cell_size = cell_size
        self.net = None

        # Downloading elevation of region from api
        elev.clip(bounds=self.bounds_latlon, output=CWD+elevation_file)

        # Getting coordinates of water bodies
        '''
        water_latlon = ox.geometries_from_place(query=place, tags={"natural": "water"}) 
        water_xy = []
        for water_geometry in water_latlon["geometry"]:
            water_xy.append(list(map(project_point, np.transpose(water_geometry.exterior.xy))))
        self.water = water_xy
        '''
        
        # Obtaining bound position in cartesian coordinates
        output_raster = CWD+"/teste.tif"
        heightmap = reproject_raster(CWD+elevation_file, output_raster)

        xy1 = (heightmap.bounds.left, heightmap.bounds.bottom)
        xy2 = (heightmap.bounds.right, heightmap.bounds.top)
        width, height = xy2[0]-xy1[0], xy2[1]-xy1[1]  


        # Building grid
        self.x, self.y = xy1
        self.lines = np.abs(int(height/cell_size))
        self.columns = np.abs(int(width/cell_size))
        
        # Initializing cells 
        # TO DO: set cells as water if inside water polygon
        self.cells = np.empty((self.lines, self.columns), dtype=Cell)

        # Getting elevation for cells
        cell_coords_v = []
        cell_idx_v = []
        for i in range(self.lines):
            for j in range(self.columns):
                cell_coords = (self.x + 1*(j*cell_size + cell_size/2), 
                               self.y + 1*(i*cell_size + cell_size/2))
                cell_coords_v.append(cell_coords)
                cell_idx_v.append((i,j))
        cell_elevation_list = list(heightmap.sample(cell_coords_v))

        # Getting elevation slope for cells
        reshaped_elevation = np.reshape(cell_elevation_list, (self.lines, self.columns))
        slopes = np.gradient(reshaped_elevation)
        slopes = np.sqrt(slopes[0]**2 + slopes[1]**2)
        #slopes = slopes/np.max(slopes)

        '''
        fig, ax = plt.subplots(1,2)
        ax[1].imshow(slopes, cmap="gray")
        ax[0].imshow(reshaped_elevation, cmap="gray")
        fig.tight_layout()
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        plt.show()
        '''

        for i, elevation in enumerate(cell_elevation_list):
            cell_coords = cell_coords_v[i]
            cell_idx = cell_idx_v[i]
            slope = slopes[cell_idx[0],cell_idx[1]]
            self.cells[cell_idx[0],cell_idx[1]] = Cell(cell_size, cell_coords, cell_idx, elevation[0], slope) # 'Position' of the cell is set in its center
        self.width = width
        self.height = height

    def getEmptyCells(self):
        flattened_cells = self.cells.flatten()
        empty_cells = [cell.idx for cell in flattened_cells if cell.empty]
        return empty_cells

    def updateNetwork(self, net):
        self.net = net

        start = time.time()
        self._getMeshDistances() # Build edge_sets (cell neighborhoods) based on distance of each sell to each edge
        
        print(f"{self.lines*self.columns} ({self.lines}x{self.columns}) cells built in {time.time() - start}s")


    # Calculates the distance of each cell in relation to the network by finding the closest edge to its center
    def _getMeshDistances(self, n_nodes=1):
        print("Calculating cell-network distances...")
        flattened_cells = self.cells.flatten()
        cell_coords = [c.position for c in flattened_cells]
        node_coords = [n for n in list(self.net.nodes.values())]

        # Tries to reduce the searched edges by looking only at those that connect the closest nodes
        node_tree = cKDTree(node_coords) 
        _, idx = node_tree.query(cell_coords, k=n_nodes, workers=-1)
        
        selected_nodes = np.array(node_coords)[idx]
        
        for c, node in tqdm(zip(flattened_cells, selected_nodes), total=len(flattened_cells)):
            c.setMeshLink(node)
            
        return

    def plotLinks(self):
        for cell in self.cells.flatten():
            plt.plot([cell.position[0], cell.linked_node[0]], [cell.position[1], cell.linked_node[1]], color="red", linewidth=.5)
        plt.axis("equal")

    def plotNetwork(self):
        if(self.net == None):
            return
        
        for edge in self.net.edges:
            edge = [self.net.nodes[edge[0]], self.net.nodes[edge[1]]]
            plt.plot(*np.transpose(edge), color="black")
        plt.axis("equal")

    def plotHMap(self):
        elevations = [c.elevation for c in self.cells.flatten()]
        elevations = np.array(elevations-min(elevations))/(max(elevations) - min(elevations))
        cells = np.zeros((self.lines, self.columns))
        for i in range(self.lines):
            for j in range(self.columns):
                color = elevations[i*self.columns + j]
                cells[-i-1,j] = color
        plt.imshow(cells, cmap="Greens", interpolation="nearest", extent=[self.x, self.x+self.width, self.y, self.y+self.height])
        plt.axis("equal")

    def plotAgents(self):
        cells = []
        for i in range(self.lines):
            cells.append([])
            for j in range(self.columns):
                cells[i].append(0)
                
        for i in range(self.lines):
            for j in range(self.columns):
                color = np.array(self.cells[i,j].type_color_rgb)
                cells[-i-1][j] = color
        cells = np.array(cells)
        plt.imshow(cells, interpolation="nearest", extent=[self.x, self.x+self.width, self.y, self.y+self.height])

    def plotPrices(self):
        elevations = [c.mesh_distance for c in self.cells.flatten()]
        elevations = np.array(elevations-min(elevations))/(max(elevations) - min(elevations))
        cells = np.zeros((self.lines, self.columns))
        for i in range(self.lines):
            for j in range(self.columns):
                color = elevations[i*self.columns + j]
                cells[-i-1,j] = color
        plt.imshow(cells, cmap="gray", interpolation="nearest", extent=[self.x, self.x+self.width, self.y, self.y+self.height])
        plt.axis("equal")

    def plotWater(self):
        for w in self.water:
            plt.plot(*np.transpose(w), color="blue")