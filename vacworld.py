#!/usr/bin/python3

# vacworld.py

#   Graphical simulation environment for VacWorld Agents according to
#   our scoring function. Each action requires 1/15 of a second to
#   execute.
#
#   See the bottom of the file for help in running your agent

import sys
import os
import random
import time
import pickle
import threading
from multiprocessing import Process, Queue
from queue import Empty

import graphics as gr
import gridutil as gu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"  # Turn off TF chatter
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Turn off GPU usage


dashes = "------------------------------"

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

    def reset(self, randomstart=False):
        self.restorePercept(self.init_percept)
        if randomstart:
            locations = [(x, y) for x in range(self.size) for y in range(self.size)
                         if (x, y) in self.reachable]
            self.agentLoc = random.choice(locations)
            self.agentDir = random.choice("NESW")

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

    color = "#EE7C00"
    # color = "red1"
    
    def __init__(self, state, height=800, title="Vacuum World", panel=False):
        self.aspect = aspect = 1.33 if panel else 1.0
        win = gr.GraphWin(title, aspect*height, height, autoflush=False)
        self.win = win
        win.setCoords(0, 0, aspect*100, 100)
        win.setBackground("azure")

        self.cells = None
        self.configureCells(state.size)

        self.vac = None
        ccenter = 116.7
        self.time = gr.Text(gr.Point(ccenter, 75), "Time").draw(win)
        self.time.setSize(36)
        self.setTimeColor("black")

        self.agentName = gr.Text(gr.Point(ccenter, 50), "").draw(win)
        self.agentName.setSize(20)
        self.agentName.setFill(self.color)

        self.info = gr.Text(gr.Point(ccenter, 25), "").draw(win)
        self.info.setSize(20)
        self.info.setFace("courier")

        self.update(state)

    def configureCells(self, xySize):
        self.xySize = xySize
        self.cellSize = 100/xySize
        if self.cells:
            for cell in self.cells.values():
                cell.undraw()
        csize = self.cellSize
        cells = {}
        for x in range(xySize):
            for y in range(xySize):
                cells[(x, y)] = gr.Rectangle(gr.Point(x*csize, y*csize),
                                             gr.Point((x+1)*csize, (y+1)*csize)
                                             )
                cells[(x, y)].setWidth(2)
                cells[(x, y)].draw(self.win)
        self.cells = cells

    def setAgent(self, name):
        self.agentName.setText(name)

    def setTime(self, seconds):
        self.time.setText(str(seconds))

    def setInfo(self, info):
        self.info.setText(info)

    def update(self, state):
        # View state in existing window
        if state.size != self.xySize:
            # this env has new size, redraw grid
            self.configureCells(state.size)
            
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
        csize = self.cellSize
        p1 = gr.Point((x+.5+dx0)*csize, (y+.5+dy0)*csize)
        p2 = gr.Point((x+.5+dx1)*csize, (y+.5+dy1)*csize)
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

    def is_alive(self):
        return self.process.is_alive()

    def quit(self):
        self.process.terminate()


# ----------------------------------------------------------------------
class ThreadAgent:

    class RequestThread(threading.Thread):

        def __init__(self, agt, percept):
            self.agt = agt
            self.percept = percept
            self.action = None
            threading.Thread.__init__(self)

        def run(self):
            self.action = self.agt(self.percept)
            sys.stdout.flush()

    def __init__(self, agent):
        self.agent = agent

    def requestAction(self, percept):
        self.thread = self.RequestThread(self.agent, percept)
        self.thread.start()

    def getAction(self):
        if self.thread.is_alive():
            return None
        else:
            action = self.thread.action
            self.thread = None
            return action

    def is_alive(self):
        return self.thread is not None

    def quit(self):
        pass


# ----------------------------------------------------------------------
class TimedSimulation:

    infostr = "{:}\n\n\n Count: {:<7d}\n\n Score: {:<7d} "

    def __init__(self, env, timelimit, win=None, use_thread=True):
        self.env = env
        self.limit = timelimit
        self.win = win or VacView(env, height=1000, panel=True)
        self.use_thread = use_thread
        self.reset()

    def reset(self):
        self.env.reset()
        self.elapsed = 0
        self.actioncount = 0
        self.score = 0
        self.lastaction = None
        self.win.setTimeColor("Black")
        self.updateAnimationView()

    def updateAnimationView(self):
        self.win.update(self.env)

    def updateInfoView(self):
        self.win.setTime(self.limit-int(self.elapsed))
        self.win.setInfo(self.infostr.format(str(self.lastaction),
                                             self.actioncount,
                                             self.score))

    def start(self):
        input("Press <Enter> to start run ")
        #print("Click to start...")
        # self.win.pause()
        self.win.setTimeColor("green")
        self.starttime = time.time()

    def still_running(self):
        return self.env.running and self.elapsed < self.limit

    def finish(self):
        self.win.setTimeColor("red")
        time_bonus = int(max(self.limit - self.elapsed+1, 0))
        self.score = self.score + time_bonus
        self.updateInfoView()
        #self.win.pause()
        #self.win.close()
        print(dashes)
        print("Actions done:", self.actioncount)
        print("Think time:", max(round(self.elapsed-self.actioncount/15, 1), 0.0))
        print("Time Bonus:", time_bonus)
        print("Final score:", self.score)

    def run(self, agent, agtName):
        print(dashes)
        agent = ProcAgent(agent) if not self.use_thread else ThreadAgent(agent)
        self.win.setAgent(agtName.upper())
        self.reset()
        self.updateInfoView()
        self.start()
        agent.requestAction(self.env.getPercept())
        while self.env.running:
            if not agent.is_alive():
                self.elapsed = self.limit+.1
                break
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
            if self.elapsed < self.limit:
                self.elapsed = time.time() - self.starttime
            self.updateInfoView()
        agent.quit()
        self.updateInfoView()
        self.finish()
        return self.score


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
    env = VacWorldEnv(size, dirt, furn)
    env.reachable = reachable
    return env


def loadEnv(fname, basedir="./envs/"):
    """ reload a saved environment """
    with open(basedir+fname, "rb") as envfile:
        env = pickle.load(envfile)
    return env


def quickSim(agt, env,  timelimit=120, report=True, trace=False):
    """Run a quick simulation

        Does not expend time executing actions so a simulation can be
        run quickly to get an approximate score. As long as the agent
        does not exceed the time limit, the returned score should be
        farily accurate.

    """
    if type(agt) == str:
        agt = loadAgent(agt, env.size, timelimit)
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
        print(dashes)
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


def test(agent, agtname="test", env=None, size=10, timelimit=120):
    """perform a simulation with the given agent or load from module agtname

    """
    env = env or randomEnv(size)
    sim = TimedSimulation(env, timelimit, )
    if type(agent) == str:
        agtname = agent
        agent = loadAgent(agent, env.size, timelimit)
    sim.run(agent, agtname)
