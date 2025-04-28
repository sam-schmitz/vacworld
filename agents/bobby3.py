#bobby3.py
#By: Sam Schmitz

from tensorflow.keras import models
import netrl
import vacworld as vw

class VacAgent:
    
    actions = "forward turnleft turnright".split()
    
    def __init__(self, size, timelimit):
        assert size <= 10
        self.size = size
        
        #load the matching keras file 
        modelfile = __name__.replace(".", "/")+".keras"
        self.net = models.load_model(modelfile)
        
        self.plan = []
        
    def __call__(self, percept):        
        #unpack percept
        location, orientation, dirt, furniture = percept
        #print(location, orientation, dirt, furniture)
                
        #if there's no dirt and agent is home return "poweroff"
        if location == (0, 0) and len(dirt) == 0:
            return "poweroff"
        #if agent location is dirty return "suck
        elif location in dirt:
            return "suck"        

        if len(self.plan) == 0:
            #run multiple sims to create the best possible plan
            while True:
                vwenv = vw.VacWorldEnv(self.size, dirt, furniture)
                netenv = netrl.NavNetEnv(vwenv)
                step_limit = 200
                k = 15
                ep = netrl.best_of_k_sims(self.net, netenv, k, step_limit)
                if len(ep) != 0:
                    break
            self.plan = ep
            print(f"Length of plan: {len(self.plan)}")            
        
        #Use the premade plan
        action = self.plan.pop(0)
        #print(f"action: {action}")
        return self.actions[action[1]]                                              



