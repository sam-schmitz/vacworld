# vac_rl.py
#    Potentially useful functions for getting information about agents
#    running in actual vac environments.

import math
import vacworld as vw


class SpyAgent:

    """Wrapper class for an agent to an record an episode while running a
    simulation. By default, it's limited to monitoring navigation
    actions.

     episode: [(percept0, action_i), (percept1, action_i), ...]

    """

    actions = ("forward", "turnleft", "turnright")

    def __init__(self, agent):
        self.agent = agent
        self.episode = []

    def __call__(self, percept):
        action = self.agent(percept)
        if action in self.actions:
            self.episode.append((percept, self.actions.index(action)))
        return action


def get_episode(agent, env, timelimit=30):
    """ Run a simulation of agent on environment and
    return the recorded episode.

    """

    if type(agent) == str:
        agent = vw.loadAgent(agent, env.size, timelimit)
    spy = SpyAgent(agent)
    env.reset()
    vw.quickSim(spy, env, timelimit, report=False)
    return spy.episode


# ----------------------------------------------------------------------
# Comparison functions
#
#   agent_list can mix objects (for reuseable, stateless agents)
#      and strings for those that need to be reloaded (e.g., "ripper").
#
#    e.g. vac_cmp_scores(["ripper", mynetagent])

def vacs_cmp_scores(agent_lst, trials=50, envsize=10, timelimit=30):
    """ Head-to-head average score over trials number of random envs
    """
    n = len(agent_lst)
    scores = [0] * n
    for _ in range(trials):
        env = vw.randomEnv(envsize)
        for agt_i in range(n):
            agent = agent_lst[agt_i]
            score = vw.quickSim(agent, env, timelimit, report=False)
            scores[agt_i] += score
            env.reset()
    return [s/trials for s in scores]


def vacs_cmp_wins(agent_lst, trials=50, envsize=10, timelimit=30):
    """Head-to-head win percentage base on score"""
    n = len(agent_lst)
    wins = [0] * n
    for i in range(trials):
        print(f"{i}:", end=" ")
        env = vw.randomEnv(envsize)
        best_score = -math.inf
        for agt_i in range(n):
            agent = agent_lst[agt_i]
            score = vw.quickSim(agent, env, timelimit, report=False)
            if score > best_score:
                best_score = score
                best_i = agt_i
            env.reset()
        wins[best_i] += 1
    return [w/trials for w in wins]


def vacs_cmp_lengths(agent_lst, trials=50, size=10):
    """Head-to-head average navigation episode length"""
    n = len(agent_lst)
    stepcounts = [0] * n
    for _ in range(trials):
        env = vw.randomEnv(size)
        for agt_i in range(n):
            agent = agent_lst[agt_i]
            steps = len(get_episode(agent, env))
            stepcounts[agt_i] += steps
            env.reset()
    return [s/trials for s in stepcounts]
