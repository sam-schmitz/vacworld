#reinforcement_learning.py
#By: Sam Schmitz
#Runs a reinforcement learning alg on a vacnet agent

import vacworld as vw
import netrl
import vacrl
import encode_percept
import pickle
import train_model
from tensorflow.keras import models
import random

MODELFILE = "agents/bobby3.keras"
TRAIN_DATA_PATH = "datasets/data_best_of_10.pk"
CHECKPOINT_PATH = "agents/bobby3_chk.keras"
MODELFILE_FINAL = "agents/bobby3.keras"
MODELFILE_CHECKPOINT = "agents/bobby3_check.keras"

def episode_iterator(net, k, num_envs):
    for _ in range(num_envs):
        vwenv = vw.randomEnv(10)
        netenv = netrl.NavNetEnv(vwenv)
        ripper_episode = vacrl.get_episode("ripper", vwenv)
        step_limit = len(ripper_episode)
        net_episode = netrl.best_of_k_sims(net, netenv, k, step_limit)
        solved = len(net_episode) < step_limit
        gain = len(ripper_episode) - len(net_episode)
        if solved and (gain > 0):
            best_ep = net_episode
        else:
            best_ep = [(encode_percept.encode_percept(10, percept), action) for percept, action in ripper_episode]
            gain = 0
        yield best_ep, gain
        
def episode_iterator_v2(net, k, num_envs, n_simul):
    vwenvs = [vw.randomEnv(10) for _ in range(n_simul)]
    net_envs = [netrl.NavNetEnv(e) for e in vwenvs]
    rip_eps = [vacrl.get_episode("ripper", ve) for ve in vwenvs]
    rip_worst = max(rip_eps, key=len)
    step_limit = len(rip_worst) + 1
    net_groups = netrl.run_k_sims(net, net_envs, k, step_limit)
    net_eps = [min(g, key=len) for g in net_groups]
    for rip, net in zip(rip_eps, net_eps):
        solved = len(net) < step_limit
        gain = len(rip) - len(net)
        if solved and gain > 0:
            best_ep = net
        else:
            best_ep = [(encode_percept.encode_percept(10, percept), action) for percept, action in rip]
            gain = 0
        yield best_ep, gain
        
def fill_buffer(net, k):
    wins = 0
    episodes = []
    for i, (ep, gain) in enumerate(episode_iterator_v2(net, k, 1000, 1000)):
        #print(i+1, len(ep), gain)
        wins += (gain > 0)
        episodes.append(ep)
    
    #split data into a validation set
    random.shuffle(episodes)
    split_index = int(len(episodes) * .95)
    train_episodes = episodes[:split_index]
    val_episodes = episodes[split_index:]
    
    #flatten the data
    train_steps = netrl.flatten(train_episodes)
    val_steps = netrl.flatten(val_episodes)    
    
    #zip the data back together
    xs_train, ys_train = zip(*train_steps)
    xs_val, ys_val = zip(*val_steps)
    
    #all_steps = netrl.flatten(episodes)
    #images, labels = zip(*all_steps)
    with open(TRAIN_DATA_PATH, "wb") as outfile:
        #pickle.dump((images, labels), outfile)
        pickle.dump({
            "train": (xs_train, ys_train), 
            "val": (xs_val, ys_val)
            }, outfile)
    print(f"wins: {wins}")
    #could add a validation set
    #

def train_on_buffer(net):
    #load the training data
    print("Loading data")
    train_data = train_model.load_data(TRAIN_DATA_PATH)
    
    #train the model
    print("starting training")
    train_model.train(net, train_data, CHECKPOINT_PATH)
    
    print("\nRestoring best model")
    return models.load_model(CHECKPOINT_PATH)    


def learn():
    #load the model
    print("Loading model")
    net = models.load_model(MODELFILE)
        
    while True:
        #save a copy of the old model for testing later
        old_net = models.clone_model(net)
        old_net.set_weights(net.get_weights())
    
        #perform the learning
        for _ in range(15):
            #fill the buffer with new training data
            print("filling buffer")
            fill_buffer(net, 15)
        
            #train the model on the data
            print("training model")
            net = train_on_buffer(net)                    
        
        #test the old and new models
        while True:
            vwenv = vw.randomEnv(10)
            netenv = netrl.NavNetEnv(vwenv)
            netenv2 = netenv
            step_limit = 500
            k = 10
            new_model_ep = netrl.best_of_k_sims(net, netenv, k, step_limit)
            old_model_ep = netrl.best_of_k_sims(old_net, netenv2, k, step_limit)
            if len(new_model_ep) != 0:
                break
            print(len(new_model_ep))
        
        #Save the better model as a checkpoint
        if len(new_model_ep) < len(old_model_ep):
            print("model improved")
        
            print("\nSaving final model:", MODELFILE_FINAL)
            net.save(MODELFILE_FINAL)
        else:
            print("model did not improve")
            print(f"new model: {len(new_model_ep)}")
            print(f"old mode: {len(old_model_ep)}")
            #net.save(MODELFILE_FINAL)
            break
        
        
def learn2():
    #load the model
    print("Loading model")
    net = models.load_model(MODELFILE)
    
    #save a copy of the old model for testing later
    old_net = models.clone_model(net)
    old_net.set_weights(net.get_weights())
    
    #save a copy of the model to a checkpoint
    net.save(MODELFILE_CHECKPOINT)

    #records how many times in a row the model has failed to improve
    model_fails = 0
    
    while True:        
    
        #perform the learning
        for _ in range(3):
            #fill the buffer with new training data
            print("filling buffer")
            fill_buffer(net, 10)
        
            #train the model on the data
            print("training model")
            net = train_on_buffer(net)  
            
        # == Check to see if the model has improved ==

        #get testing data
        while True:
            vwenv = vw.randomEnv(10)
            netenv = netrl.NavNetEnv(vwenv)
            netenv2 = netenv.copy()
            step_limit = 500
            k = 10
            new_model_ep = netrl.best_of_k_sims(net, netenv, k, step_limit)
            old_model_ep = netrl.best_of_k_sims(old_net, netenv2, k, step_limit)
            if len(new_model_ep) != 0:
                break
            print(len(new_model_ep))
        
        #check for improvement
        if len(new_model_ep) < len(old_model_ep):
            print("model improved")
            model_fails = 0
            
            #save a checkpoint
            net.save(MODELFILE_CHECKPOINT)
            
            #change old net to the current best net
            old_net = models.clone_model(net)
            old_net.set_weights(net.get_weights())
        else:
            print("model did not improve")
            model_fails += 1
            
            #if the model has failed to much load the best so far and break
            if model_fails > 5:
                net = models.load_model(MODELFILE_CHECKPOINT)
                break
            
    #save the new model
    print("saving model")
    net.save(MODELFILE_FINAL)  
        
if __name__ == "__main__":
    learn2()    
