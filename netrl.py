# netrl

#    Basic environment for running simulations on navnets for the
#    vacworld environment and some useful building blocks for
#    reinforcement learning.

from collections import deque
import pickle
import random
import os

import numpy as np
from tensorflow.keras import models

import vacworld as vw
import vacrl
from encode_percept import encode_percept

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Turn off TF chatter

class NavNetEnv:
    """Quick and dirty stripped down vacworld environment for training
    navigation networks in the vacworld

    It provides observations as numpy arrays (10, 10, 6) and accepts
    actions as ints in range(3).

    """
    forward_offset = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(self, vwenv=None):
        vwenv = vwenv or vw.randomEnv(10)
        self.vwenv = vwenv
        self.init_percept = vwenv.getPercept()
        self.state = np.zeros((10, 10, 6), dtype=bool)
        self.reset()

    def reset(self):
        # Code moved to separate file for handout
        percept = self.init_percept
        self.state = encode_percept(self.vwenv.size, percept)

    def agentinfo(self):
        return np.argwhere(self.state[:, :, :4] == 1)[0]

    def dirtmap(self):
        return self.state[:, :, 4]

    def furnmap(self):
        return self.state[:, :, 5]

    def getObs(self):
        return self.state.copy()

    def doAction(self, action):
        if self.done():
            return
        state = self.state
        x, y, ori = np.argwhere(state[:, :, :4] == 1)[0]
        state[x, y, ori] = 0
        if action == 0:  # forward
            dx, dy = self.forward_offset[ori]
            x1 = x + dx
            y1 = y + dy
            if 0 <= x1 < 10 and 0 <= y1 < 10 and not state[x1, y1, 5]:
                x, y = x1, y1
            state[x, y, ori] = 1
            state[x, y, 4] = 0  # any visited space is cleaned

        elif action == 1:  # turnleft
            ori1 = (ori - 1) % 4
            state[x, y, ori1] = 1

        else:   # turnright
            ori1 = (ori + 1) % 4
            state[x, y, ori1] = 1

    def done(self):
        clean = np.all(self.state[:, :, 4] == 0)
        x, y, _ = np.argwhere(self.state[:, :, :4] == 1)[0]
        home = x == 0 and y == 0
        return clean and home

    def copy(self):
        clone = NavNetEnv(self.vwenv)
        clone.state = self.state.copy()
        return clone

    def __str__(self):
        # ascii map of current state
        grid = np.array([["."]*10 for _ in range(10)])
        grid[self.dirtmap().astype(bool)] = "O"
        grid[self.furnmap().astype(bool)] = "X"
        x, y, ori = self.agentinfo()
        achar = "^>V<"[ori]
        grid[x, y] = achar
        # to get x,y view, need to transpose and reverse axis 0
        grid = grid.T[::-1, :]
        rows = [" ".join(r) for r in grid]
        return "\n".join(rows)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.state == other.state


def runsims(net, envlist, steplimit=200, trace=False):
    """ Use net to run simultaneous simulations

    This is the main function for running simulations. Other
    variations below rely on this one to do the work.

    net: vw image (10, 10, 6) --> prob. dist (3,)
    envlist:  list of NavNetEnv objects to use net on

    returns: a list of "episodes" (one per env in envlist)
             an episode is a list of steps (state, action_index)
             state is of shape (10, 10, 6)
             action_index in range(3)

    """
    # episodes is the list of lists where steps will be added
    episodes = [list() for e in envlist]

    # live_envs and live_episodes are parallel lists tracking live sims
    live_envs = [e for e in envlist if not e.done()]
    if not live_envs:  # bail out if no steps required to solve any episodes
        return episodes
    live_episodes = [lst for lst, e in zip(episodes, envlist) if not e.done()]

    for i in range(steplimit):
        # collect the observations and send them through the net
        observations = [e.getObs() for e in live_envs]
        inputs = np.array(observations)
        outputs = net.predict(inputs, verbose=0)

        # turn net policy outputs into actions
        actions = [np.random.choice(3, p=net_out) for net_out in outputs]

        # do the actions!
        for env, action in zip(live_envs, actions):
            env.doAction(action)
            if trace:
                print("action:", action)
                print(env)
                input("pause")

        # update live_episodes and live_envs
        for obs, action, steplist in zip(observations, actions, live_episodes):
            steplist.append((obs, action))
        live_pairs = [(e, s) for e, s in zip(live_envs, live_episodes)
                      if not e.done()]
        if not live_pairs:
            break
        live_envs, live_episodes = zip(*live_pairs)
    return episodes


def runsim(net, env, steplimit):
    """Run vanilla simulation of net in env"""
    return runsims(net, [env], steplimit)[0]


def run_k_sims(net, envlist, k=3, steplimit=400):
    """run k simulations of each environment

    returns a list containing k episodes for each environment.
    [
     [env0_episode0, env0_episode1, ...],
     [env1_episode0, env1_episode1, ...],
     ...
    ]

    """
    num_envs = len(envlist)
    allenvs = []
    for _ in range(k):
        for e in envlist:
            allenvs.append(e.copy())
    episodes = runsims(net, allenvs, steplimit)
    episode_groups = []
    for offset in range(num_envs):
        group = episodes[offset::num_envs]
        episode_groups.append(group)
    return episode_groups


def best_of_k_sims(net, env, k=50, steplimit=200):
    envlst = [env.copy() for _ in range(k)]
    episodes = runsims(net, envlst, steplimit)
    print(f"NUmber of episodes: {len(episodes)}")
    for i, ep in enumerate(episodes):
        print(f"    Episode {i}: length={len(ep)}")
    return min(episodes, key=len)


def flatten(listoflists):
    """ Combine all items from listoflists

    Useful to combine the steps of a bunch of episodes.

    """
    return [item for sublist in listoflists for item in sublist]


# function to determine percentage of net wins over ripper for
# various values of sims_per_env. Not really all that useful,
# but shows how to mix vacworld and navnet environments.

def net_wins_vs_ripper(net, sims_per_env, num_envs=50):
    vwenvs = [vw.randomEnv(10) for _ in range(num_envs)]
    netenvs = [NavNetEnv(env) for env in vwenvs]

    ripper_lengths = [len(vacrl.get_episode("ripper", env))
                      for env in vwenvs]
    steplimit = max(ripper_lengths)

    net_episode_groups = run_k_sims(net, netenvs, sims_per_env, steplimit)
    net_lengths = [len(min(g, key=len)) for g in net_episode_groups]

    wins = 0
    for r, n in zip(ripper_lengths, net_lengths):
        if n < r:  # Careful here. Tie probably means net sim was incomplete
            wins += 1
    return wins/num_envs


class EpisodeBuffer:

    """ Buffer for episodes to draw training batches from

    To get entire buffer as a training set: buff.get_batch(buff.n_steps)

    """
    def __init__(self, capacity):
        # capacity is number of episodes saved
        self.episodes = deque([], maxlen=capacity)

    def add(self, episode):
        self.episodes.append(episode)

    def size(self):
        return len(self.episodes)

    def get_batch(self, batch_size):
        """ returns a random batch of steps
        """
        all_steps = flatten(self.episodes)
        step_sample = random.sample(all_steps, batch_size)
        states, actions = zip(*step_sample)
        states = np.array(states, dtype="int32")
        actions = np.array(actions, dtype="uint8")
        return states, actions

    @property
    def n_steps(self):
        """number or total steps across all episodes
        """
        return len(flatten(self.episodes))

    def save(self, filename):
        """Save this buffers as a list of epiosodes"""
        with open(filename, "wb") as outfile:
            pickle.dump(list(self.episodes), outfile)

    def reload(self, filename):
        """add episodes in filename to this buffer"""
        with open(filename, "rb") as infile:
            self.episodes.extend(pickle.load(infile))


if __name__ == "__main__":     # a bit of testing code
    buffer = EpisodeBuffer(10)
    rapper = models.load_model("agents/rapper.keras")
    envs = [NavNetEnv() for i in range(12)]
    episodes = runsims(rapper, envs)
    for e in episodes:
        buffer.add(e)
