# jimmy.py
# billy, but break problem into subchunks
# By: Sam Schmitz & Andrew Poock

import gridutil
import math
import search
from datetime import datetime, timedelta

class VacAgent:

    actions = "turnleft turnright forward".split()

    def __init__(self, size, timelimit=120):
        self.steps = 0
        self.route = []
        self.size = size
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(seconds=timelimit)
        self.dirty_chunks = [] # Keep track of chunks with dirt inside
        self.chunk_size = math.floor(math.sqrt(size))

    def __call__(self, percept):
        location, orientation, dirt, furniture = percept
        self.update_chunks(dirt) # update dirty chunks

        if location == (0, 0) and len(dirt) == 0:
            return "poweroff"
        elif location in dirt:
            return "suck"
        #Added a new go home method
        elif(self.end_time - datetime.now()) < timedelta(seconds=7):   
            self._go_home(location, orientation, furniture)
        #moved this check
        elif len(dirt) == 0:
            self._go_home(location, orientation, furniture)
        
        if len(self.route) == 0:
            # find target dirt using chunks instead of closest
            d = find_target_dirt(location, dirt, self.chunk_size, self.dirty_chunks)
            prob = dirtFindingProblem(location, d, orientation, furniture, self.size)
            result = search.astar_search(prob)  #switch to astar vs idastar
            self.route = result.extract_plan()
            
        return self.route.pop(0)
    
    def _go_home(self, location, orientation, furniture):
        if gridutil.manhatDist(location, (0,0)) > 10:
            #check if bot should go in x, y or both directions
            goX, goY = True, True
            if not gridutil.legalLoc((location[0]-5, location[1]), self.size):
                target = (location[0], location[1]-5)
                goX = False
            elif not gridutil.legalLoc((location[0], location[1]-5), self.size):
                target = (location[0]-5, location[1])
                goY = False
            else:
                target = (location[0]-5, location[1]-5)
                    
            #find a target location without furniture
            while target not in furniture:                    
                newTarget = target
                if goX:
                    newTarget = (newTarget[0]+1, newTarget[1])
                    if newTarget not in furniture:
                        target = newTarget 
                        break
                if goY:
                    newTarget = (newTarget[0], newTarget[1]+1)
                    if newTarget not in furniture:
                        target = newTarget
                        break
                target = newTarget

            prob = dirtFindingProblem(location, target, orientation, furniture, self.size)
        else:
            #close enough to home that direct search is possible
            prob = dirtFindingProblem(location, (0,0), orientation, furniture, self.size)
        result = search.astar_search(prob)  #note astar search
        self.route = result.extract_plan()
        #add poweroff so bot doesn't go in circles
        self.route.append("poweroff")

    def update_chunks(self, dirt):
        dirty_chunks = {(x // self.chunk_size, y // self.chunk_size) for x, y in dirt}

        rows = {}
        for ch in dirty_chunks: # Group chunks by row (y val)
            rows.setdefault(ch[1], []).append(ch)

        # Sort each row: l to r for even, r to l for odd
        sorted_chunks = []
        for row in sorted(rows.keys()):
            row_chunks = sorted(rows[row], key=lambda ch: ch[0])
            if row % 2 == 1:  # Reverse odd rows
                row_chunks.reverse()
            sorted_chunks.extend(row_chunks)

        self.dirty_chunks = sorted_chunks
    
    
class dirtFindingProblem:
    
    def __init__(self, start, dirtLoc, ori, furn, n):
        self.initial_state = (start, ori, dirtLoc)
        self.furn = furn
        self.size = n
        
    def successors(self, state):
        loc, ori, dirt = state
        loc1 = gridutil.nextLoc(loc, ori)
        if gridutil.legalLoc(loc1, self.size) and (loc not in self.furn):
            yield("forward", (loc1, ori, dirt))
        yield("turnleft", (loc, gridutil.leftTurn(ori), dirt))
        yield("turnright", (loc, gridutil.rightTurn(ori), dirt))
    
    def goal_test(self, state):
        return state[0] == state[2]

    #added a hueristic
    def h(self, state):
        return gridutil.manhatDist(state[0], state[2])
    
    #cost = {"turnleft": 20, "turnright": 20, "forward": 20}
    

def findClosestDirt(agentLoc, dirt):
    closest = math.inf
    closestTile = None
    for tile in dirt:
        tileDist = gridutil.manhatDist(agentLoc, tile)
        if tileDist == 1:
            return tile
        elif tileDist < closest:
            closest = tileDist
            closestTile = tile
    return closestTile

def find_target_dirt(loc, dirt, chunk_size, dirty_chunks):
        current_chunk = (loc[0] // chunk_size, loc[1] // chunk_size)
        chunk_dirt = [d for d in dirt if (d[0] // chunk_size, d[1] // chunk_size) == current_chunk]
        if chunk_dirt: # if there is dirt in current chunk, go clean nearest one
            return findClosestDirt(loc, chunk_dirt)
        next_chunk = dirty_chunks.pop(0) # If no dirt in current chunk, go to next chunk in snake order
        chunk_dirt = [d for d in dirt if (d[0] // chunk_size, d[1] // chunk_size) == next_chunk]
        return findClosestDirt(loc, chunk_dirt)

def validLoc(loc, size, furn):
    return gridutil.legalLoc(loc, size) and (loc in furn)