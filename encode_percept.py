# encode_percept.py

import numpy as np


def encode_percept(n, percept):
    (x, y), ori, dirt, furn = percept
    state = np.zeros((10, 10, 6), dtype="uint8")

    # set agent
    channel = "NESW".index(ori)
    state[x, y, channel] = 1

    # set dirt
    if dirt:
        xs, ys = zip(*dirt)
        state[xs, ys, 4] = 1
        state[0, 0, 4] = 0    # reflex suck clears home

    # set furniture
    if furn:
        xs, ys = zip(*furn)
        state[xs, ys, 5] = 1

    # set boundary
    if n < 10:
        state[:n, n, 5] = 1
        state[n, :n, 5] = 1

    return state
