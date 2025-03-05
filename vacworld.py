#!/usr/bin/python3

# vacworld.py

#   Graphical simulation environment for VacWorld Agents according to
#   our scoring function. Each action requires 1/15 of a second to
#   execute.
#
#   See the bottom of the file for help in running your agent


import random
import time
import pickle
from multiprocessing import Process, Queue
from queue import Empty

import graphics as gr
import gridutil as gu


class VacWorldEnv:
    """
    Scoring:
    
        Each action costs: 20 point
        Reward per cell cleaned: 100
        Penalty for bumping walls or furniture: 100
        Bonus for being home at end of simulation: 500
        Penalty for NOT being home at end: 10 * manhattan distance from home
        Bonus for clearing all dirt: 500
        Bonus for remaining time: 1 point per second

    """

    actions = "suck turnleft turnright forward poweroff".split()

    def __init__(self, size, dirt, furniture):
        self.init_percept = ((0, 0), "E", dirt, furniture)
        self.size = size
        self.reset()

    def reset(self):
        self.restorePercept(self.init_percept)

    def restorePercept(self, p):
        loc, orient, dirt, furn = p
        self.agentLoc = loc
        self.agentDir = orient
        self.dirt = set(dirt)
        self.furniture = set(furn)
        self.running = True

    def getPercept(self):
        return (self.agentLoc,
                self.agentDir,
                tuple(self.dirt),
                tuple(self.furniture))

    def doAction(self, action):
        points = -20
        if action == "turnleft":
            self.agentDir = gu.leftTurn(self.agentDir)
        elif action == "turnright":
            self.agentDir = gu.rightTurn(self.agentDir)
        elif action == "forward":
            loc = gu.nextLoc(self.agentLoc, self.agentDir)
            if gu.legalLoc(loc, self.size) and loc not in self.furniture:
                self.agentLoc = loc
            else:
                points -= 100  # "scuff" penalty
        elif action == "suck":
            if self.agentLoc in self.dirt:
                self.dirt.remove(self.agentLoc)
                points += 100  # clean up reward
        elif action == "poweroff":
            self.running = False
            if self.agentLoc == (0, 0):
                points += 500  # got home bonus
            else:
                # not home penalty
                points -= 10*gu.manhatDist(self.agentLoc, (0, 0))
            if not self.dirt:
                points += 500  # job finished bonus
        return points

    def show(self):
        return VacView(self, height=800)

    def save(self, filename):
        with open(filename, "wb") as envfile:
            pickle.dump(self, envfile)


class VacView:

    # VacView shows a view of a VacWinEnv. Just hand it an env, and
    #   a window will pop up.

    VacSize = .2
    VacPoints = {'N': (0, -VacSize, 0, VacSize),
                 'E': (-VacSize, 0, VacSize, 0),
                 'S': (0, VacSize, 0, -VacSize),
                 'W': (VacSize, 0, -VacSize, 0)}

    color = "orange"

    def __init__(self, state, height=800, title="Vacuum World", panel=False):
        xySize = state.size
        aspect = 1.33 if panel else 1.0
        win = gr.GraphWin(title, aspect*height, height, autoflush=False)
        self.win = win
        win.setCoords(-.5, -.5, aspect*xySize-.5, xySize-.5)
        win.setBackground("gray99")

        cells = self.cells = {}
        for x in range(xySize):
            for y in range(xySize):
                cells[(x, y)] = gr.Rectangle(gr.Point(x-.5, y-.5),
                                             gr.Point(x+.5, y+.5))
                cells[(x, y)].setWidth(2)
                cells[(x, y)].draw(win)
        self.vac = None
        ccenter = 1.167*(xySize-.5)
        self.time = gr.Text(gr.Point(ccenter, (xySize-1)*.75), "Time").draw(win)
        self.time.setSize(36)
        self.setTimeColor("black")

        self.agentName = gr.Text(gr.Point(ccenter, (xySize-1)*.5), "").draw(win)
        self.agentName.setSize(20)
        self.agentName.setFill("Orange")

        self.info = gr.Text(gr.Point(ccenter, (xySize-1)*.25), "").draw(win)
        self.info.setSize(20)
        self.info.setFace("courier")

        self.update(state)

    def setAgent(self, name):
        self.agentName.setText(name)

    def setTime(self, seconds):
        self.time.setText(str(seconds))

    def setInfo(self, info):
        self.info.setText(info)

    def update(self, state):
        # View state in exiting window
        for loc, cell in self.cells.items():
            if loc in state.dirt:
                cell.setFill("lightgray")
            elif loc in state.furniture:
                cell.setFill("black")
            else:
                cell.setFill("white")

        if self.vac:
            self.vac.undraw()

        x, y = state.agentLoc
        dx0, dy0, dx1, dy1 = self.VacPoints[state.agentDir]
        p1 = gr.Point(x+dx0, y+dy0)
        p2 = gr.Point(x+dx1, y+dy1)
        vac = gr.Line(p1, p2)
        vac.setWidth(5)
        vac.setArrow('last')
        vac.setFill(self.color)
        vac.draw(self.win)
        self.vac = vac

    def pause(self):
        self.win.getMouse()

    def setTimeColor(self, c):
        self.time.setTextColor(c)

    def close(self):
        self.win.close()


# ----------------------------------------------------------------------
class ProcAgent:
    """Wrapper to run an agent as a separate process"""

    def __init__(self, agentprog):
        self.agent = agentprog
        self.perceptQ = Queue()
        self.actionQ = Queue()
        self.process = Process(target=self.run,
                               args=(self.perceptQ, self.actionQ))
        self.process.start()

    def run(self, perceptQ, actionQ):
        while True:
            percept = perceptQ.get()
            action = self.agent(percept)
            actionQ.put(action)

    def requestAction(self, percept):
        self.perceptQ.put(percept)

    def getAction(self):
        try:
            action = self.actionQ.get(False)
        except Empty:
            action = None
        return action

    def quit(self):
        self.process.terminate()


# ----------------------------------------------------------------------
class TimedSimulation:

    infostr = "{:}\n\n\n Count: {:<7d}\n\n Score: {:<7d} "

    def __init__(self, env, timelimit, win=None):
        self.env = env
        self.limit = timelimit
        self.win = win or VacView(env, height=600, panel=True)
        self.reset()

    def reset(self):
        self.env.reset()
        self.elapsed = 0
        self.actioncount = 0
        self.score = 0
        self.lastaction = None
        self.updateAnimationView()

    def updateAnimationView(self):
        self.win.update(self.env)

    def updateInfoView(self):
        self.win.setTime(self.limit-int(self.elapsed))
        self.win.setInfo(self.infostr.format(str(self.lastaction),
                                             self.actioncount,
                                             self.score))

    def start(self):
        self.win.pause()
        self.win.setTimeColor("green")
        self.starttime = time.time()

    def still_running(self):
        return self.env.running and self.elapsed < self.limit

    def finish(self):
        self.win.setTimeColor("red")
        time_bonus = int(max(self.limit - self.elapsed+1, 0))
        self.score = self.score + time_bonus
        self.updateInfoView()
        self.win.pause()
        self.win.close()        
        print("--------------------------------------------")
        print("Actions done:", self.actioncount)
        print("Think time:", max(round(self.elapsed-self.actioncount/15, 1), 0.0))
        print("Time Bonus:", time_bonus)
        print("Final score:", self.score)

    def run(self, agent, agtName):
        agent = ProcAgent(agent)
        self.win.setAgent(agtName.upper())
        self.reset()
        self.updateInfoView()
        self.start()
        agent.requestAction(self.env.getPercept())
        while self.env.running:
            if self.elapsed >= self.limit:
                # pull the plug
                action = "poweroff"
            else:
                action = agent.getAction()
            if action:
                self.score += self.env.doAction(action)
                if action != "poweroff":
                    # send request for next action
                    agent.requestAction(self.env.getPercept())
                self.actioncount += 1
                self.lastaction = action
                self.updateAnimationView()
            gr.update(15)
            self.elapsed = time.time() - self.starttime
            self.updateInfoView()
        agent.quit()
        self.updateInfoView()
        self.finish()


# ----------------------------------------------------------------------
# Useful testing functions

def randomEnv(size=4, dprob=.3, fprob=.15):
    """return a random environment with the given specs

    """
    furn = [loc for loc in gu.locations(size)
            if loc != (0, 0) and random.random() < fprob]

    # create set of all reachable locations
    reachable = set()
    frontier = [(0, 0)]
    while frontier:
        loc = frontier.pop()
        reachable.add(loc)
        for direction in gu.DIRECTIONS:
            neighbor = gu.nextLoc(loc, direction)
            if (gu.legalLoc(neighbor, size)
                and neighbor not in furn
                and neighbor not in reachable
                ):
                frontier.append(neighbor)

    dirt = [loc for loc in reachable if random.random() < dprob]
    return VacWorldEnv(size, dirt, furn)


def loadEnv(fname, basedir="./envs/"):
    """ reload a saved environment """
    with open(basedir+fname, "rb") as envfile:
        env = pickle.load(envfile)
    return env


def quickSim(agentname, env, timelimit=120, report=True, trace=False):
    """Run a quick simulation

        Does not expend time executing actions so a simulation can be
        run quickly to get an approximate score. As long as the agent
        does not exceed the time limit, the returned score should be
        farily accurate.

    """
    agt = loadAgent(agentname, env.size, timelimit)
    env.reset()
    score = 0
    step = 1
    max_steps = timelimit*15
    start = time.time()
    while env.running:
        percept = env.getPercept()
        action = agt(percept) if step < max_steps else "poweroff"
        score = score + env.doAction(action)
        if trace:
            print("Step:", step)
            print("percept:", percept)
            print("action:", action)
            print("score:", score, "\n")
        step = step + 1
    stop = time.time()
    elapsed = stop-start
    bonus = int(timelimit - elapsed - (step-1)/15 + 1)
    score = score + bonus
    if report:
        print("--------------------------------------------")
        print("elapsed time:", round(elapsed, 2))
        print("steps:", step-1)
        print("time bonus", bonus)
        print("final score:", score)
    return score


def loadAgent(modname, n, timelimit):
    """return agent from the given module file initialized with size n
    Use this function to create an object from your agent file. For
    example, if you have a class VacAgent in the file myagent.py,
    simply call loadAgent("myagent", 8) to get an agent object for an
    env of size 8.

    """
    mod = __import__("agents."+modname, fromlist=[''])
    agent = mod.VacAgent(n, timelimit)
    return agent


def test(agtname, agent=None, env=None, size=10, timelimit=120):
    """perform a simulation with the given agent or load from module agtname

    """
    env = env or randomEnv(size)
    sim = TimedSimulation(env, timelimit)
    agent = agent or loadAgent(agtname, env.size, timelimit)
    sim.run(agent, agtname)
