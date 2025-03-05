# runsim.py

"""Script to simplify running tests on agents

To get quick help with the options, type:

   $ python3 runsim.py -h

The following command runs the agent in myagent.py on a random 10x10
env with 120 seconds; the env is also saved into a file (lastsim.env),
overwriting the previous contents.

  $ python3 runsim.py myagent

To replay the last simulation, just tack on a -r switch.

   $ python3 runsim.py myotheragent -r

If you want save an environment for later, just copy the lastsim.env
into the envs folder and rename it (e.g. hard10x10.env)

Then you can use it again later using the -e switch:

   $ python3 runsim.py myagent -e hard10x10.env

You can also choose a random environment with different sizes and
time limits.  For example you could create a simulation of size 20
and a 60 second time limit like so:

    $ python3 runsim.py myagent -s 20 -t 60

Finally, you can get a "quick simulation" by adding the -q flag. See
the quickSim function in vacworld for ore information on what this does.

    $ python3 runsim.py myagent -s 20 -q

"""

import argparse
import vacworld as vw
from vacworld import VacWorldEnv  # need for pickling


DEFAULT_FILE = "lastsim.env"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("agtname",
                        help="name of VacAgent module to run (w/o .py)")
    parser.add_argument("-r", "--replay", help="run with last environment",
                        action="store_true")
    parser.add_argument("-q", "--quick", help="run a quick simulation",
                        action="store_true")
    parser.add_argument("-s", "--size", type=int, default=10,
                        help="generate random env of this size")
    parser.add_argument("-n", "--nosave", help="don't save env into lastsim.env",
                        dest="saveenv", action="store_false")
    parser.add_argument("-e", "--env", type=str,
                        help="use env from supplied file")
    parser.add_argument("-t", "--time", type=int, default=120,
                        help="time limit for simulation")
    args = parser.parse_args()
    print()

    # create an env
    if args.replay:
        env = vw.loadEnv(DEFAULT_FILE, basedir="./")
    elif args.env:
        print(f"Using environment {args.env}")
        env = vw.loadEnv(args.env)
    else:
        env = vw.randomEnv(args.size, dprob=.3)

    print("Grid Size:", env.size)
    print("Dirty:", len(env.dirt))
    print("--------------------------------------------")

    # unless told not to, save the env
    if args.saveenv:
        env.save(DEFAULT_FILE)

    # do either quicksim or full timed run
    if args.quick:
        vw.quickSim(args.agtname, env, args.time)
    else:
        sim = vw.TimedSimulation(env, args.time)
        agent = vw.loadAgent(args.agtname, env.size, args.time)
        sim.run(agent, args.agtname)


if __name__ == "__main__":
    main()
