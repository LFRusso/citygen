import matplotlib.pyplot as plt

from citygen import generate
from citygen.plot import plotRoad

EPISODES = 5
place = 'Quadra, São Paulo, Brazil'
generator = generate(place)

def plotEpisode(world):
    world.plotHMap()
    world.plotAgents()
    world.plotNetwork()
    #world.plotLinks()
    #world.plotWater()
    plt.tight_layout()
    plt.axis("off")
    plt.show()

def plotPrices(world):
    world.plotHMap()
    world.plotAgents()
    world.plotPrices()
    world.plotNetwork()
    plt.tight_layout()
    plt.axis("off")
    plt.show()

for i in range(EPISODES):
    print("episode", i+1)
    generator.runEpisode()

    plotEpisode(generator.world)
    plotPrices(generator.world)