import json
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import descartes
import geopandas as gpd
from matplotlib.patches import Polygon
import osmnx as ox
import elevation
import rasterio
from rasterio.plot import show
import os
from geopy.distance import geodesic,distance
import math
from pyproj import Transformer
import utm

from mapgen import generate
from grid import Grid
from plot import plotRoad, plotGraph
from world import World

def plotSegments(generator):
    edges = generator.segments
    lines = [(edge.start, edge.end) for edge in edges if True]
    artists_contour = []
    for verts in lines:
        x, y = zip(*verts)
        plt.plot(x, y, color='red')        


#### Generate road and grid
'''
SEED = 42
np.random.seed(SEED)

MAX_ITER = 500
width, height = 2000, 2000
generator = generate(0, 0, width, height, seed=SEED)
grid = Grid(0,0, width, height, 50)

for i in range(MAX_ITER):
    generator.step()
grid.updateNetwork(generator.graph)

print("plotting road")
plotRoad(generator.segments)
print("plotting grid")
grid.plotGrid(links=False)
plt.show()
'''
####

#### Get geo information about region

place = 'Quadra, São Paulo, Brazil'
#place = 'Tsukuba, Ibaraki, Japan'
CWD = os.getcwd()
elevation_file = '/data/elevation.tif'
world = World(place, elevation_file, cell_size=200)

####

#### Integrate geo information with road generation

SEED = 42
np.random.seed(SEED)

MAX_ITER = 0
width, height = world.width, world.height
x, y = world.x, world.y
generator = generate(x, y, width, height, seed=SEED)
for i in range(MAX_ITER):
    generator.step()

world.plotHMap()
#plotSegments(generator)
#plotRoad(generator.segments)
plt.show()

####


'''
with rasterio.open(CWD+elevation_file) as rds:
    data = rds.read()
plt.imshow(data[0], cmap='gray')
plt.show()


with rasterio.open(CWD+"/teste.tif") as rds:
    data = rds.read()
plt.imshow(data[0], cmap='gray')
plt.show()
'''

'''
CWD = os.getcwd()
SEED = 42
np.random.seed(SEED)
MAX_ITER = 1000
with open("config.json") as config_file:
    config = json.load(config_file)

# Getting height map and water
place = 'Tsukuba, Ibaraki, Japan'

land = ox.geocode_to_gdf(place)
water = ox.geometries_from_place(query=place, tags={"natural": "water"})
bounds = water.total_bounds
#elevation.clip(bounds=bounds, output=CWD+'/data/quadra.tif')
rds = rasterio.open("data/quadra.tif")



pos1 = bounds[:2][::-1]
pos2 = bounds[2:][::-1]

x = utm.from_latlon(*pos1)[:2]
y = utm.from_latlon(*pos2)[:2]
print(x)
print(y)

print(x[0] - y[0], x[1] - y[1])
width, height = latlon2wh(pos1, pos2)
print(width, height)
#print(list(rds.sample([(x, y)])))

def aux(latlon):
    lat, lon = latlon
    return utm.from_latlon(lat, lon)


fig, ax = ox.plot_footprints(water,
                             color="blue", bgcolor="white",
                             show=False, close=False)
ax = water.plot(ax=ax, fc="blue", markersize=0)
plt.show()

for w in water["geometry"]:
    xx, yy = w.exterior.xy
    cartesian_xy = list(map(aux, zip(yy, xx)))
    #plt.plot(*np.transpose(cartesian_xy), color="blue")
    plt.plot(*w.exterior.xy, color="blue")
plt.axis("equal")
plt.show()
'''

#####

'''
fig, ax = ox.plot_footprints(water,
                             color="blue", bgcolor="white",
                             show=False, close=False)
ax = water.plot(ax=ax, fc="blue", markersize=0)
plt.show()
pos1 = bounds[:2]
pos2 = bounds[2:]

x = 0
y = 0

#print(np.abs(xy1[0] - xy2[0]), np.abs(xy1[1] - xy2[1]))
#print(xy1[0] - xy1[1], xy2[0] - xy2[1])
#width, height = latlon2wh(pos1, pos2)
#print(width, height)
for w in water["geometry"]:
    xx, yy = w.exterior.xy
    cartesian_xy = list(map(latlon2xy, zip(xx, yy)))
    #plt.plot(*np.transpose(cartesian_xy), color="blue")
    plt.plot(*w.exterior.xy, color="blue")
plt.axis("equal")
plt.show()
for w in water["geometry"]:
    xx, yy = w.exterior.xy
    cartesian_xy = list(map(latlon2xy, zip(xx, yy)))
    plt.plot(*np.transpose(cartesian_xy), color="blue")
plt.axis("equal")
plt.show()


with rasterio.open("data/quadra.tif") as rds:
    data = rds.read()
    transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)
    #xx, yy = transformer.transform(lon, lat)
    print(rds.sample([(0, 0)]))
print(transformer.transform(0,0))
#print(latlon2wh(pos1, pos2))

#print(data.shape)
#plt.imshow(data[0], cmap='gray')
#plt.show()

'''
'''
generator = generate(0, 0, width, height, seed=SEED)
grid = Grid(x, y, width, height, 50)

for i in range(MAX_ITER):
    generator.step()
grid.updateNetwork(generator.graph)

print("plotting road")
plotRoad(generator.segments)
print("plotting grid")
grid.plotGrid(links=True)
plt.show()
'''
