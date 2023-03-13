import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

MARKETS = [
    {'id':1, 'wage': 100, 'product': 20},
    {'id':2, 'wage': 200, 'product': 30},
    {'id':3, 'wage': 300, 'product': 40}
]

class Agents:
    def __init__(self, new_agents = 10, new_markets = 2):
        self.agents = []
        self.markets = []
        self.new_agents = new_agents
        self.new_markets = new_markets

        self.v0 = 13
        self.v_foot = 1.5
        return

    def exploration(self, world):
        print("exploring")
        empty_cells = world.getEmptyCells()
        idx = np.random.choice(range(len(empty_cells)), self.new_agents+self.new_markets)
        new_devels = np.array(empty_cells)[idx]

        # Adds new markets
        for i in range(self.new_markets):
            position = new_devels[i]
            cell = world.cells[position[0], position[1]]
            market = Market(cell)
            self.markets.append(market)

            cell.setDeveloped(2+market.type) # Setting as a market


        # Adds new agents
        for i in range(self.new_markets, len(new_devels)):
            position = new_devels[i] 
            cell = world.cells[position[0], position[1]]
            agent = Agent(cell)
            self.agents.append(agent)
            # Adding agent as worker of a random new market
            idx = np.random.randint(len(self.markets))
            market = self.markets[idx]
            market.addWorker(agent)

            cell.setDeveloped(1) # Setting as a residence

        # Update prices
        t = self.travelTime(world.cells[0,0], world.cells[-1,-1], world.net)
        print(t, "seconds")
        return

    # Finds travel trime using shortest path between two cells in the map by A*
    def travelTime(self, start_cell, end_cell, net):
        def dist(a, b):
            (x1, y1) = G.nodes[a]["pos"]
            (x2, y2) = G.nodes[b]["pos"]
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            t = d/self.v0
            return t

        # No network built yet, use euclidean distance
        dist_cells = np.linalg.norm(np.array(start_cell.position)-np.array(end_cell.position))
        if (net==None):
            t = dist_cells / self.v_foot
            return t

        # If distance to the network is greater than distance to the end cell, do not use network
        dist_net_in = np.linalg.norm(np.array(start_cell.position)-np.array(start_cell.linked_node))
        dist_net_out = np.linalg.norm(np.array(end_cell.position)-np.array(end_cell.linked_node))
        if (dist_net_in >= dist_cells or dist_net_out >= dist_cells):
            print("cells are close")
            t = dist_cells / self.v_foot
            return t

        edges = list(net.edges.keys())
        G = nx.Graph()
        G.add_edges_from(edges)

        nx.set_node_attributes(G, {n: net.nodes[n] for n in G.nodes()}, 'pos')
        nx.set_edge_attributes(G, {e: dist(e[0],e[1]) for e in G.edges()}, "cost")
        
        start_node = net.findNodeId(start_cell.linked_node)
        end_node = net.findNodeId(end_cell.linked_node)
        
        path_len = nx.astar_path_length(G, start_node, end_node, heuristic=dist, weight="cost")
        print("time in net:", path_len)

        total_time = dist_net_in/self.v_foot + path_len + dist_net_out/self.v_foot
        return total_time

        #path = nx.astar_path(G, start_node, end_node, heuristic=dist, weight="cost")
        #path_edges = list(zip(path, path[1:]))
        #pos = {n:G.nodes[n]["pos"] for n in G.nodes()}
        #nx.draw(G, pos, node_color="k", node_size=5)
        #nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r')
        #plt.show()

        return

    # Simulates the cycle of a day for the agents
    def runDay(self):
        for agent in self.agents:
            agent.planDay()
        
        for agent in self.agents:
            agent.step(self.agent)

    # Adds n new agents to the system, initialized randomly
    def addAgents(self, n):
        return


class Agent:
    def __init__(self, cell):
        self.x, self.y = cell.idx
        self.residence = cell
        self.job = None
        self.money = 5000

    def setJob(self, market):
        self.job = market

    # Chooses what the agent will do during this day; returns path to be taken
    def planDay(self):
        return

    # Takes a step according to the current plan, its outcome depending on the positions and actions of other agents
    def step(self, other_agents):
        return

class Market:
    def __init__(self, cell, market_type=None):
        self.cell = cell
        self.x, self.y = cell.idx
        self.workers = []

        if (market_type != None):
            self.type = market_type
        else: 
            # Selecting a random market type
            self.type = np.random.randint(len(MARKETS))
        self.wage = MARKETS[self.type]
        self.product = MARKETS[self.type]
    
    def addWorker(self, agent):
        self.workers.append(agent)
        agent.setJob(self)
        return