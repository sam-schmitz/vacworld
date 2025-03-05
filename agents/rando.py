# rando.py
#  simple example agent showing the basic form for an agent file.
# updated by: Sam and Poock

import random
import time


class VacAgent:

    actions = "turnleft turnright forward".split()

    def __init__(self, size, timelimit=120):
        self.steps = 0

    def __call__(self, percept):
        time.sleep(random.random()*.5)  # pretend thinking

        # example of how to unpack the percept
        location, orientation, dirt, furniture = percept

        if location == (0, 0) and len(dirt) == 0:
            return "poweroff"
        elif location in dirt:
            return "suck"

        return random.choice(self.actions)
