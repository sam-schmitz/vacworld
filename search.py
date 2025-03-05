# search.py

from collections import deque
import math
import heapq

class SearchNode:
    """A SearchNode encapsulates a problem state produced during a
    state-space search. In addition to the problem state, it also
    records a reference to the parent node and the action that lead
    from the parent state to this state.

    """

    def __init__(self, state, parent=None, action=None, step_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = parent.depth + 1 if parent else 0
        self.path_cost = step_cost + (self.parent.path_cost if parent else 0)
        
    def make_child(self, action, state, step_cost=None):
        """returns a child node of self having the given action and state"""
        return SearchNode(state, parent=self, action=action, step_cost=step_cost)

    def expansion(self, problem):
        """generates the successor nodes of self"""
        for action, resultstate in problem.successors(self.state):
            step_cost = problem.cost[action] if hasattr(problem, "cost") else 1
            yield self.make_child(action, resultstate, step_cost)

    def extract_plan(self):
        """returns the list of actions that led to this node"""
        steps = []
        curr = self
        while curr.action is not None:
            steps.append(curr.action)
            curr = curr.parent
        steps.reverse()
        return steps

    def __repr__(self):
        return f"SearchNode(state={self.state}, action={self.action})"

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)


def bf_tree_search(problem):
    """perform a breadth-first tree search.

    problem is a search problem
    if successful, returns the goal node from the search

    """
    frontier = deque([SearchNode(problem.initial_state)])
    while len(frontier) > 0:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        new_nodes = node.expansion(problem)
        frontier.extend(new_nodes)


def bf_graph_search(problem):
    """perform a breadth-first graph search"""
    frontier = deque([SearchNode(problem.initial_state)])
    explored = set()
    while len(frontier) > 0:
        node = frontier.popleft()
        if node and problem.goal_test(node.state):
            return node
        explored.add(node.state)
        new_nodes = [n for n in node.expansion(problem)
                     if n not in frontier and n.state not in explored]
        frontier.extend(new_nodes)


def df_tree_search(problem):
    """depth-first tree search using an explicit stack"""
    frontier = [SearchNode(problem.initial_state)]
    while len(frontier) > 0:
        node = frontier.pop()
        if node and problem.goal_test(node.state):
            return node
        new_nodes = list(node.expansion(problem))
        frontier.extend(new_nodes)


def rdf_search(problem):
    """recursive depth-first search"""

    def rdfs(problem, currnode):
        if problem.goal_test(currnode.state):
            return currnode
        for nextnode in currnode.expansion(problem):
            result = rdfs(problem, nextnode)
            if result:
                return result

    return rdfs(problem, SearchNode(problem.initial_state))


def rdfcp_search(problem):
    """recursive depth-first search with cyle prevention

    """
    path_states = set()

    def rdfs(problem, currnode):
        if problem.goal_test(currnode.state):
            return currnode
        for nextnode in currnode.expansion(problem):
            if nextnode and nextnode.state not in path_states:
                path_states.add(nextnode.state)
                result = rdfs(problem, nextnode)
                if result:
                    return result
                path_states.remove(nextnode.state)

    return rdfs(problem, SearchNode(problem.initial_state))


def db_search(problem, maxdepth):
    """recursive depth-bounded search

    Only considers paths of length <= maxdepth.
    """

    def dbs(problem, currnode, depth_allowed):
        if problem.goal_test(currnode.state):
            return currnode
        if depth_allowed == 0:
            return None
        for nextnode in currnode.expansion(problem):
            result = dbs(problem, nextnode, depth_allowed-1)
            if result:
                return result

    return dbs(problem, SearchNode(problem.initial_state), maxdepth)


def id_search(problem, fail_depth=501, print_level=False, db_alg=db_search):
    """iterative deepening search

    performs depth-bounded searches with increasing depth bounds to
    guarantee an optimal solution.

    """
    for depth in range(1, fail_depth):
        if print_level:
            print(depth)
        result = db_alg(problem, depth)
        if result:
            return result


def dbcp_search(problem, maxdepth):
    """depth-bounded search with cycle checking/avoidance"""

    path = set()

    def dbs(problem, currnode, depth_allowed):
        if problem.goal_test(currnode.state):
            return currnode
        if depth_allowed == 0:
            return None
        for nextnode in currnode.expansion(problem):
            if nextnode.state not in path:
                path.add(nextnode.state)
                result = dbs(problem, nextnode, depth_allowed-1)
                if result:
                    return result
                path.remove(nextnode.state)

    return dbs(problem, SearchNode(problem.initial_state), maxdepth)


def idcp_search(problem, fail_depth=501, print_level=True):
    """iterative deepening search with cycle checking/avoidance"""
    return id_search(problem, fail_depth, print_level, dbcp_search)


_ida_incr = math.inf


def idastar_search(problem, maxdepth=500):
    """iterative deepening with heuristic pruning

    """
    global _ida_incr
    path = set()

    def dbs(currnode, depth_allowed):
        global _ida_incr
        if problem.goal_test(currnode.state):
            return currnode
        hval = problem.h(currnode.state)
        if depth_allowed < hval:
            _ida_incr = min(hval-depth_allowed, _ida_incr)
            return None
        for nextnode in currnode.expansion(problem):
            if nextnode.state not in path:
                path.add(nextnode.state)
                cost = (problem.cost[currnode.action]
                        if hasattr(problem, "cost") else 1)
                result = dbs(nextnode, depth_allowed-cost)
                if result:
                    return result
                path.remove(nextnode.state)

    state = problem.initial_state
    depth = problem.h(state)
    while depth <= maxdepth:
        # print(depth)
        _ida_incr = math.inf
        result = dbs(SearchNode(problem.initial_state), depth)
        if result is not None:
            return result
        depth = depth + _ida_incr

    return None


def astar_search(problem):
    """perform a breadth-first graph search"""

    frontier = PriorityQueue()
    fval = problem.h(problem.initial_state)
    frontier.push_or_update(SearchNode(problem.initial_state), fval)
    explored = set()
    while len(frontier) > 0:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for node in node.expansion(problem):
            fvalue = node.path_cost + problem.h(node.state)
            if node.state not in explored:
                frontier.push_or_update(node, fvalue)
            

class PriorityQueue:
    def __init__(self):
        self._heap = []
        self._count = 0  # To handle items with the same priority
        self._entry_finder = {}  # Mapping of items to their heap entries

    def push_or_update(self, item, priority):
        if item in self._entry_finder:
            entry = self._entry_finder[item]
            if entry[0] <= priority:
                return
            self.remove(item)
        entry = [priority, self._count, item]
        self._entry_finder[item] = entry
        heapq.heappush(self._heap, entry)
        self._count += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from an empty priority queue")
        while self._heap:
            priority, count, item = heapq.heappop(self._heap)
            if item is not None:
                del self._entry_finder[item]
                return item
        raise IndexError("Pop from an empty priority queue")

    def is_empty(self):
        return not bool(self._entry_finder)

    def remove(self, item):
        if item in self._entry_finder:
            entry = self._entry_finder.pop(item)
            entry[-1] = None  # Mark as removed
    
    def __len__(self):
        return len(self._entry_finder)

    def clear(self):
        self._heap = []
        self._entry_finder.clear()
        self._count = 0


if __name__ == "__main__":
    from searchprob import RouteFindingProblem
    prob = RouteFindingProblem("A", "R")
    result = bf_tree_search(prob)
    print("Final State:", result.state)
    print("Plan:", result.extract_plan())
