# ripper: a streamlined greedy vac
# by John Zelle

from collections import deque
import gridutil as gu


class VacAgent:

    def __init__(self, n, timelimit=120):
        self.n = n
        self.plan = []
        self.planner = None

    def __call__(self, percept):
        loc, ori, dirt, furn = percept
        self.planner = self.planner or Planner(self.n, furn)

        if not self.plan:
            if dirt:
                # find some dirt and clean it up
                self.plan = self.planner.get_route(loc, ori, dirt)
                self.plan.insert(0, "suck")
            else:
                # go home
                #print("Going Home")
                self.plan = self.planner.get_route(loc, ori, [(0, 0)])
                self.plan.insert(0, "poweroff")

        return self.plan.pop()


class Planner:
    """Simple breadth-first graph search"""

    actions = {"f": "forward", "l": "turnleft", "r": "turnright"}

    def __init__(self, n, avoid):
        self.n = n
        self.avoid = avoid

    def get_route(self, loc, ori,  dests):
        node = ((loc, ori), "")
        frontier = deque([node])
        explored = set()
        while frontier:
            node = frontier.popleft()
            state, planstr = node
            if state[0] in dests:
                plan = [self.actions[ch] for ch in planstr]
                return plan
            explored.add(state)
            new_nodes = [n for n in self.expand_node(node)
                         if n[0] not in explored]
            frontier.extend(new_nodes)
        return None

    def expand_node(self, node):
        (loc, ori), planstr = node
        loc1 = gu.nextLoc(loc, ori)
        if gu.legalLoc(loc1, self.n) and loc1 not in self.avoid:
            yield ((loc1, ori), "f" + planstr)
        yield ((loc, gu.leftTurn(ori)), "l" + planstr)
        yield ((loc, gu.rightTurn(ori)), "r" + planstr)
