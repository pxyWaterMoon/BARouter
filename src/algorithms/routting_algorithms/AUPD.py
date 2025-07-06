from src.algorithms.routting_algorithms.base_model import OnlineModel
from src.algorithms.predictor.xgb import XGB
from src.logger import Logger
import numpy as np

class AUPD(OnlineModel):
    def __init__(self, rmodel:XGB, cmodel:XGB, logger:Logger, T, budget, embedding_fn, buffer_size = 1024):
        self.budget = budget
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.b = self.budget / T
        self.Q = 0
        self.V = self.b * np.sqrt(T)

        self.X_buffer = []
        self.r_buffer = []
        self.c_buffer = []
        self.buffer_size = buffer_size
        self.t = 0
        self.logger = logger
        self.embedding_fn = embedding_fn
    
    def take_action(self, sample):
        if self.embedding_fn is None:
            raise ValueError("Embedding function is not provided.")
        X = self.embedding_fn(sample)
        action_space = list(sample["available_models_description"].keys())
        # print(X.shape)
        # X: (K, d)
        predict_reward:np.ndarray = self.rmodel.predict(X) # (K)
        predict_cost:np.ndarray = self.cmodel.predict(X) # (K)
        # print(predict_cost.shape)
        weight = predict_reward - (self.Q/self.V)*predict_cost # (K)
        action_index = np.argmax(weight)
        action = action_space[action_index]
        self.logger.log_scalar(
            {
                "train/predict_reward": predict_reward[action_index],
                "train/predict_cost": predict_cost[action_index],
            },
            step=self.t,
        )
        # print(action)
        self.X_buffer.append(X[action_index])
        return action
    
    def update(self, reward, cost):
        self.r_buffer.append(reward)
        self.c_buffer.append(cost)
        self.Q = max(self.Q + cost - self.budget,0)
        self.logger.log_scalar(
                {
                    "train/Q": self.Q
                },
                step=self.t,
            )
        self.t += 1
        if len(self.X_buffer) > self.buffer_size:
            self.rmodel.online_update(np.array(self.X_buffer),np.array(self.r_buffer))
            self.cmodel.online_update(np.array(self.X_buffer),np.array(self.c_buffer))
            self.X_buffer = []
            self.r_buffer = []
            self.c_buffer = []
            
