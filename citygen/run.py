import matplotlib.pyplot as plt

from __init__ import generate
from plot import plotRoad

EPISODES = 1
place = 'Tatuí, São Paulo, Brazil'
generator = generate(place)

def plotEpisode(world):
    #world.plotHMap()
    #world.plotAgents()
    world.plotPrices()
    world.plotNetwork()
    #world.plotLinks()
    #world.plotWater()
    plt.tight_layout()
    plt.axis("off")
    plt.show()

for i in range(EPISODES):
    print("episode", i+1)
    generator.runEpisode()

    plotEpisode(generator.world)

#plotRoad(generator.roadnet.segments)
#plt.show()