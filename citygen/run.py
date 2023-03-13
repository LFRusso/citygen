import matplotlib.pyplot as plt

from __init__ import generate
from plot import plotRoad

EPISODES = 1

place = 'Quadra, São Paulo, Brazil'
generator = generate(place)

for i in range(EPISODES):
    generator.runEpisode()

#plotRoad(generator.roadnet.segments)
#generator.world.plotHMap()
#generator.world.plotNetwork()
#plt.show()