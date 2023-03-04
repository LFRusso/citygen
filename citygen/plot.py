import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import geopandas as gpd
from matplotlib.patches import Polygon

def plotGraph(graph, x_lim, y_lim):
    plt.rcParams['axes.facecolor']='#f5f7f8'
    plt.rcParams['savefig.facecolor']='#f5f7f8'

    for edge in graph.edges.keys():
        x, y = np.transpose((edge[0], edge[1]))
        line_width = 2 if graph.edges[edge]["highway"] else 3
        #plt.plot(x, y, color=graph.edges[edge]["color"], linewidth=line_width)
        line_color = "#fde293" if graph.edges[edge]["highway"] else "#ffffff"
        plt.plot(x, y, color="#adb1ca", linewidth=line_width+.1)
        plt.plot(x, y, color=line_color, linewidth=line_width)
    
    plt.axis('scaled')
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.show()

def _plotMinorRoads(edges, ax):
    # Plotting contours
    lines = [(edge.start, edge.end) for edge in edges if edge.highway == False]
    artists_contour = []
    for verts in lines:
        x, y = zip(*verts)
        line, = ax.plot(x, y, color='#dadce0')
        artists_contour.append(line)

    # Plotting middle
    artists = []
    for verts in lines:
        x, y = zip(*verts)
        line, = ax.plot(x, y, color='white')
        artists.append(line)

    scalar_countour = StrokeScalar(artists_contour, 13)
    scalar = StrokeScalar(artists, 7)

    ax.callbacks.connect('xlim_changed', scalar_countour)
    ax.callbacks.connect('ylim_changed', scalar_countour)
    ax.callbacks.connect('xlim_changed', scalar)
    ax.callbacks.connect('ylim_changed', scalar)
   
def _plotHighways(edges, ax):
    # Plotting contours
    lines = [(edge.start, edge.end) for edge in edges if edge.highway == True]
    artists_contour = []
    for verts in lines:
        x, y = zip(*verts)
        line, = ax.plot(x, y, color='#f9ad05')
        artists_contour.append(line)

    # Plotting middle
    artists = []
    for verts in lines:
        x, y = zip(*verts)
        line, = ax.plot(x, y, color='#fde293')
        artists.append(line)

    scalar_countour = StrokeScalar(artists_contour, 35)
    scalar = StrokeScalar(artists, 20)

    ax.callbacks.connect('xlim_changed', scalar_countour)
    ax.callbacks.connect('ylim_changed', scalar_countour)
    ax.callbacks.connect('xlim_changed', scalar)
    ax.callbacks.connect('ylim_changed', scalar)

def plotRoad(edges):
    plt.rcParams['axes.facecolor']='#e1e1e1'
    plt.rcParams['savefig.facecolor']='white'
    fig, ax = plt.subplots()

    _plotMinorRoads(edges, ax)
    _plotHighways(edges, ax)

    # Rescale things to leave a bit of room around the edges...
    ax.margins(0.05) 
    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    plt.axis('scaled')


# Implementation shared in https://stackoverflow.com/a/15673254
class StrokeScalar(object):
    def __init__(self, artists, width):
        self.width = width
        self.artists = artists
        # Assume there's only one axes and one figure, for the moment...
        self.ax = artists[0].axes
        self.fig = self.ax.figure

    def __call__(self, event):
        """Intended to be connected to a draw event callback."""
        for artist in self.artists:
            artist.set_linewidth(self.stroke_width)

    @property
    def stroke_width(self):
        positions = [[0, 0], [self.width, self.width]]
        to_inches = self.fig.dpi_scale_trans.inverted().transform
        pixels = self.ax.transData.transform(positions)
        points = to_inches(pixels) * 72
        return points.ptp(axis=0).mean() # Not quite correct...
