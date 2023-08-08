import matplotlib.pyplot as plt

from __init__ import generate
from plot import plotRoad

EPISODES = 1
place = 'São Carlos, São Paulo, Brazil'
generator = generate(place)

for i in range(EPISODES):
    print("episode", i+1)
    generator.runEpisode()


    generator.world.plotHMap()
    #generator.world.plotAgents()
    #generator.world.plotPrices()
    generator.world.plotNetwork()
    plt.tight_layout()
    plt.axis("off")
    plt.show()

#plotRoad(generator.roadnet.segments)
#plt.show()