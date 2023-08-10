import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

MARKETS = [
    {'id':1, 'wage': 250, 'product': 20, "expenses":100},
    {'id':2, 'wage': 350, 'product': 30, "expenses":200},
    {'id':3, 'wage': 450, 'product': 40, "expenses":300}
]

class Agents:
    def __init__(self, new_agents = 5*50, new_markets = 5*35, cycles = 10, view_radius = 5):
        self.agents = []
        self.markets = []
        self.new_agents = new_agents
        self.new_markets = new_markets
        self.cycles = cycles
        self.view_radius = view_radius

        self.v0 = 13
        self.v_foot = 1.5

        self.G = None
        return

    def dist(self, a, b):
        (x1, y1) = self.G.nodes[a]["pos"]
        (x2, y2) = self.G.nodes[b]["pos"]
        d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        t = d/self.v0
        return t

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

        
        # Building graph from road network
        edges = list(world.net.edges.keys())
        G = nx.Graph()
        G.add_edges_from(edges)
        self.G = G

        nx.set_node_attributes(self.G, {n: world.net.nodes[n] for n in self.G.nodes()}, "pos")
        nx.set_edge_attributes(self.G, {e: self.dist(e[0],e[1]) for e in self.G.edges()}, "cost")

        # Startig cycles
        for i in range(self.cycles):
            print("RUNNING CYCLE", i)
            # Agent activity
            for agent in self.agents:
                self.runDay(agent, world.net)

            # Market activity
            print()
            for market in self.markets:
                self.marketDynamics(market, world)

            # Updaring prices
            print("Recalculating cell prices...")
            flattened_cells = world.cells.flatten()
            for i in tqdm(range(len(flattened_cells))):
                self.getPrice(flattened_cells[i], world)
            print("Finished recalculating cell prices!\n")

            # Agent relocation
            print()
            for agent in self.agents:
                self.relocateAgent(agent, world)
            print()
        return

    # Given a cell, lotes the closest market to it
    def getClosestMarket(self, cell, market_type, net):
        selected_markets = [market for market in self.markets if market.type==market_type]
        if (len(selected_markets) == 0):
            # No market of selected type found
            return None, None

        pool = Pool(processes=12)
        results = [pool.apply_async(self.travelTime, [cell, market.cell, net]) for market in selected_markets]
        distances = np.zeros(len(selected_markets))
        for idx, val in enumerate(results):
            distances[idx] = val.get()
        
        idx = np.argsort(distances)

        closest = selected_markets[idx[0]]
        dist_closest = distances[idx[0]]

        return closest, dist_closest

        closest = selected_markets[0]
        dist_closest = self.travelTime(cell, closest.cell, net)
        for market in selected_markets:
            dist_new = self.travelTime(cell, market.cell, net)
            if (dist_new < dist_closest):
                closest = market
                dist_closest = dist_new
        return closest, dist_closest

    # Chooses what the agent will do during this day; simulates the cycle of a day for it
    def runDay(self, agent, net):
        current_position = agent.residence # Start position = residence position
        total_time = 0

        # 1. chooses order to visit markets of different types
        market_types = [i for i in range(len(MARKETS))]
        np.random.shuffle(market_types)

        # 2. Commute to work
        commute_time = self.travelTime(agent.residence, agent.job.cell, net)
        total_time += commute_time
        current_position = agent.job.cell

        # 3. Visits closest markets ordered by type 
        for market_type in market_types:
            market, time = self.getClosestMarket(current_position, market_types[market_type], net)
            if (market != None):
                current_position = market.cell
                agent.visit(market)
                total_time += time
        
        # 4. Returns home
        time = self.travelTime(current_position, agent.residence, net)
        total_time += time
        current_position = agent.residence

        # 5. Receives wage
        agent.money += agent.job.wage
        print("agent finished loop in", total_time,"ms")
        return

    # Checks if market is relocated based on this cycles' profit and expenses
    def marketDynamics(self, market, world):
        expenses = 1000+market.wage * len(market.workers)
        print("EXPENSES:", expenses)
        print("PROFITS:", market.cycle_profit)
        if (market.cycle_profit < expenses):
            # Relocate
            i, j = market.x, market.y
            vicinity = world.cells[max(i-self.view_radius,0) : min(i+self.view_radius+1, world.lines),
                                max(j-self.view_radius,0) : min(j+self.view_radius+1, world.columns)]
            vicinity = list(vicinity.flatten())
            vicinity = [c for c in vicinity if c.empty]
            scores = [c.price for c in vicinity]

            # Sorting cells by score
            idx = np.argsort(scores)
            scores = np.array(scores)[idx]
            vicinity = np.array(vicinity)[idx]
            new_cell = vicinity[-1]

            market.relocate(new_cell)
            print()
        else:
            print("not relocating\n")
    
    # Calculates the price of a cell give its geographical location and markets
    def getPrice(self, cell, world):
        #max_distance = max([c.mesh_distance for c in world.cells.flatten()])
        #price = 7000*np.exp(np.log(2)*cell.mesh_distance/max_distance) + np.exp(-cell.slope)*3000 # Provisional value, change after fixing threading
        price = cell.mesh_distance
        cell.price = price
        return price
        market_types = [i for i in range(len(MARKETS))]
        distances = [self.getClosestMarket(cell, market, world.net)[1] for market in market_types]
        price = 7000*sum([np.exp(-d) for d in distances])/len(MARKETS) + np.exp(-cell.slope)*3000
        
        cell.price = price
        return price


    def relocateAgent(self, agent, world):
        # Selecting agent vicinity
        i, j = agent.x, agent.y
        vicinity = world.cells[max(i-self.view_radius,0) : min(i+self.view_radius+1, world.lines),
                            max(j-self.view_radius,0) : min(j+self.view_radius+1, world.columns)]
        vicinity = list(vicinity.flatten())
        vicinity = [c for c in vicinity if c.empty]

        # Selecting 10 extra random cells
        empty_cells = world.getEmptyCells()
        idx = np.random.choice( range(len(empty_cells)), 3)
        random_cells = np.array(empty_cells)[idx]
        for cell_idx in random_cells:
            cell = world.cells[cell_idx[0], cell_idx[1]]
            vicinity.append(cell)

        # Calculating price of all cells in the vicinity
        #prices = [self.getPrice(cell, world) for cell in vicinity]
        
        # Calculating personal score for agent
        pool = Pool(processes=12)
        #results = [pool.apply_async(self.travelTime, [agent.job.cell, cell, world.net]) for cell in vicinity]
        personal_scores = np.zeros((len(vicinity)))
        #for index, val in enumerate(results):
        #    personal_scores[index] = 2 - val.get() * np.log(2) / 1000
        #personal_scores = [ 2 - np.exp(self.travelTime(agent.job.cell, cell, world.net) * np.log(2) / 1000) for cell in vicinity] 

        # Final score for the agent = price + personal score
        #scores = [p + s for p,s in zip(prices, personal_scores)]
        scores = [c.mesh_distance + s for c, s in zip(vicinity, personal_scores)]

        # Sorting cells by score
        idx = np.argsort(scores)
        scores = np.array(scores)[idx]
        vicinity = np.array(vicinity)[idx]

        # Checking if best prospect location is better than current
        current_score = agent.residence.mesh_distance #+ (2 - np.exp(self.travelTime(agent.job.cell, agent.residence, world.net) * np.log(2) / 1000))
        best_cell = vicinity[0]
        best_cell_score = scores[0] #+ (2 - np.exp(self.travelTime(agent.job.cell, best_cell, world.net) * np.log(2) / 1000))
        
        print("current:", best_cell_score)
        print("best:", best_cell_score)
        if (best_cell_score < current_score):
            agent.moveTo(best_cell)
        else:
            pass
            print("not moving")

    # Finds travel trime using shortest path between two cells in the map by A*
    def travelTime(self, start_cell, end_cell, net):
        # No network built yet, use euclidean distance
        dist_cells = np.linalg.norm(np.array(start_cell.position)-np.array(end_cell.position))
        if (net==None):
            t = dist_cells / self.v_foot
            return t

        # If distance to the network is greater than distance to the end cell, do not use network
        dist_net_in = np.linalg.norm(np.array(start_cell.position)-np.array(start_cell.linked_node))
        dist_net_out = np.linalg.norm(np.array(end_cell.position)-np.array(end_cell.linked_node))
        if (dist_net_in >= dist_cells or dist_net_out >= dist_cells):
            t = dist_cells / self.v_foot
            return t

        start_node = net.findNodeId(start_cell.linked_node)
        end_node = net.findNodeId(end_cell.linked_node)
        
        path_len = nx.astar_path_length(self.G, start_node, end_node, heuristic=self.dist, weight="cost")
        total_time = dist_net_in/self.v_foot + path_len + dist_net_out/self.v_foot
        return total_time


class Agent:
    def __init__(self, cell):
        self.x, self.y = cell.idx
        self.residence = cell
        self.job = None
        self.money = 5000

    def setJob(self, market):
        self.job = market

    # Takes a step according to the current plan, its outcome depending on the positions and actions of other agents
    def step(self, other_agents):
        return

    # Agents visit a market, spending money on the product it sells
    def visit(self, market):
        self.money -= market.product
        market.cycle_profit += market.product

    # Changes residence 
    def moveTo(self, cell):
        print("moving!")
        self.residence.setEmpty()
        self.residence = cell
        self.x, self.y = cell.idx
        cell.setDeveloped(1)

class Market:
    def __init__(self, cell, market_type=None):
        self.cell = cell
        self.x, self.y = cell.idx
        self.workers = []

        self.money = 10000
        self.cycle_profit = 0

        if (market_type != None):
            self.type = market_type
        else: 
            # Selecting a random market type
            self.type = np.random.randint(len(MARKETS))
        self.wage = MARKETS[self.type]["wage"]
        self.product = MARKETS[self.type]["product"]
    
    def addWorker(self, agent):
        self.workers.append(agent)
        agent.setJob(self)
        return

    def relocate(self, cell):
        self.cell.setEmpty()
        self.cell = cell
        self.cell.setDeveloped(2+self.type)
        self.x, self.y = self.cell.idx