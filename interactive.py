# vac_demo.py

from vacworld import randomEnv, VacView
from graphics import update
import time


class DemoAgent:

    KEYS = {"Left": "turnleft",
            "Right": "turnright",
            "Up": "forward",
            "p": "poweroff",
            "s": "suck"
            }

    def __init__(self, win):
        self.win = win

    def __call__(self, percept):
        while True:
            key = self.win.getKey()
            if key in self.KEYS:
                break
        return self.KEYS[key]


    
def main():
    env = randomEnv(5)
    view = VacView(env, panel=True)
    agent = DemoAgent(view.win)
    view.setAgent("INTERACTIVE")
    score = 0
    start = time.time()
    step = 0
    while env.running:
        percept = env.getPercept()
        action = agent(percept)
        score = score + env.doAction(action)
        step = step + 1
        view.update(env)
        view.setTime(int(120 - (time.time() - start)+1))
        view.setInfo(f"{action}\n\n\n Score: {score:<7d}\n\n Count: {step:<7d}")
        update()
    stop = time.time()
    elapsed = stop-start
    bonus = int(120 - elapsed - (step)/15 + 1)
    score = score + bonus
    view.setInfo(f"{action}\n\n\n Score: {score:<7d}\n\n Count: {step:<7d}")
    view.win.getMouse()
    print("final score:", score)


main()
