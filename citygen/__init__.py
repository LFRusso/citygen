import numpy as np

from world import World
from roadnet import RoadNet
from agents import Agents

ELEVATION_FILE = '/data/elevation.tif'
SEED = 52
MAX_ITER = 500

np.random.seed(SEED)

class Generator:
    def __init__(self, world, roadnet, agents):
        self.world = world
        self.roadnet = roadnet
        self.agents = agents

    def runEpisode(self):
        # Running development phase
        for i in range(MAX_ITER):
            self.roadnet.step()
        self.world.updateNetwork(self.roadnet.graph)

        # Running exploration phase
        #self.agents.exploration(self.world)
        #self.roadnet.updateHeatmap(self.world)


# TO DO: add support to heightmap matrix instead of geo 
def generate(place):
    # Generating grid based on selected location
    world = World(place, ELEVATION_FILE, cell_size=50)

    width, height = world.width, world.height
    x, y = world.x, world.y

    price_matrix = np.zeros((world.lines, world.columns))
    for i in range(world.lines):
        for j in range(world.columns):
            price_matrix[i,j] = world.cells[i,j].price
    roadnet = RoadNet(x, y, width, height, price_matrix)

    agents = Agents()
    return Generator(world, roadnet, agents)