from src.algorithms.online_models.base_model import OnlineModel
from src.algorithms.offline_model.xgb import XGB
import numpy as np

class AUPD(OnlineModel):
    def __init__(self, rmodel:XGB, cmodel:XGB, T, budget, buffer_size = 1024):
        self.budget = budget
        self.rmodel = rmodel
        self.cmodel = cmodel

        self.Q = 0
        self.V = budget * np.sqrt(T)

        self.X_buffer = []
        self.r_buffer = []
        self.c_buffer = []
        self.buffer_size = buffer_size
    
    def take_action(self, X):
        predict_reward:np.ndarray = self.rmodel.predict(X)
        predict_cost:np.ndarray = self.cmodel.predict(X)
        # print(predict_cost.shape)

        weight = predict_reward - (self.Q/self.V)*predict_cost
        action = np.argmax(weight)
        # print(action)
        self.Q = max(self.Q + predict_cost[action] - self.budget,0)
        return action
    
    def update(self, X, reward, cost):
        self.X_buffer.append(X)
        self.r_buffer.append(reward)
        self.c_buffer.append(cost)
        if len(self.X_buffer) > self.buffer_size:
            self.rmodel.online_update(np.array(self.X_buffer),np.array(self.r_buffer))
            self.cmodel.online_update(np.array(self.X_buffer),np.array(self.c_buffer))
            self.X_buffer = []
            self.r_buffer = []
            self.c_buffer = []
            
