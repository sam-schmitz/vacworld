# willy.py
# By: Sam Schmitz & Andrew Poock

import random
import time
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

    def __call__(self, percept):

        location, orientation, dirt, furniture = percept

        if location == (0, 0) and len(dirt) == 0:
            return "poweroff"
        elif location in dirt:
            return "suck"
        elif(self.end_time - datetime.now()) < timedelta(seconds=7):   
            self._go_home(location, orientation, furniture)
        elif len(dirt) == 0:
            self._go_home(location, orientation, furniture)
        
        if len(self.route) == 0:
            #make new route
            d = findClosestDirt(location, dirt)
                
            prob = dirtFindingProblem(location, d, orientation, furniture, self.size)
            result = search.astar_search(prob)
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
    
class dirtFindingProblem:
    
    def __init__(self, start, dirt, ori, furn, n):
        self.initial_state = (start, ori)
        self.furn = furn
        self.size = n
        self.cost = {"turnleft": 20, "turnright": 20, "forward": 20}
        self.dirt = dirt
        
        
    def successors(self, state):
        loc, ori = state
        loc1 = gridutil.nextLoc(loc, ori)
        if gridutil.legalLoc(loc1, self.size) and (loc not in self.furn):
            yield("forward", (loc1, ori))
        yield("turnleft", (loc, gridutil.leftTurn(ori)))
        yield("turnright", (loc, gridutil.rightTurn(ori)))
        
    
    def goal_test(self, state):
        return state[0] in self.dirt

    """#added a hueristic
    def h(self, state):
        return gridutil.manhatDist(state[0], state[2])
    """
    
class routeFindingProblem:
    
    def __init__(self, start, target, ori, furn, n):
        self.initial_state = (start, ori, target)
        self.furn = furn
        self.size = n
        self.cost = {"turnleft": 20, "turnright": 20, "forward": 20}
        
        
    def successors(self, state):
        loc, ori, target = state
        loc1 = gridutil.nextLoc(loc, ori)
        if gridutil.legalLoc(loc1, self.size) and (loc not in self.furn):
            yield("forward", (loc1, ori, target))
        yield("turnleft", (loc, gridutil.leftTurn(ori), target))
        yield("turnright", (loc, gridutil.rightTurn(ori), target))
        
    
    def goal_test(self, state):
        return state[0] == state[2]

    def h(self, state):
        return gridutil.manhatDist(state[0], state[2])
    

    

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

def validLoc(loc, size, furn):
    return gridutil.legalLoc(loc, size) and (loc in furn)


