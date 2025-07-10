from src.algorithms.routting_algorithms.base_model import OnlineModel
from src.algorithms.predictor.xgb import XGB
from src.logger import Logger
import numpy as np

class AUPD(OnlineModel):
    def __init__(self, rmodel:XGB, cmodel:XGB, logger:Logger, T, budget, embedding_fn, buffer_size = 1024, v_scale = 1.0, allow_null = False):
        self.budget = budget
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.b = self.budget / T
        self.Q = 0
        self.V = self.b * np.sqrt(T)

        self.V = self.V * v_scale # scale

        self.rinput_buffer = []
        self.cinput_buffer = []
        self.r_buffer = []
        self.c_buffer = []
        self.buffer_size = buffer_size
        self.t = 0
        self.logger = logger
        self.embedding_fn = embedding_fn
        self.allow_null = allow_null
    
    def take_action(self, sample):
        action_space = list(sample["available_models_description"].keys())
        
        cmodel_input = self.embedding_fn(sample, concatenate=self.cmodel.concatenate)
        # print(X.shape)
        # X: (K, d)
        if self.rmodel.concatenate:
            rmodel_input = self.embedding_fn(sample, concatenate=self.rmodel.concatenate)
            predict_reward:np.ndarray = self.rmodel.predict(rmodel_input) # (K)
        else:
            rmodel_input_x, rmodel_input_a = self.embedding_fn(sample, concatenate=self.rmodel.concatenate)
            predict_reward:np.ndarray = self.rmodel.predict(rmodel_input_x, rmodel_input_a) # (K)
        if self.cmodel.concatenate:
            cmodel_input = self.embedding_fn(sample, concatenate=self.cmodel.concatenate)
            predict_cost:np.ndarray = self.cmodel.predict(cmodel_input) # (K)
        else:
            cmodel_input_x, cmodel_input_a = self.embedding_fn(sample, concatenate=self.cmodel.concatenate)
            predict_cost:np.ndarray = self.cmodel.predict(cmodel_input_x, cmodel_input_a) # (K)
        # print(predict_cost.shape)
        weight = predict_reward - (self.Q/self.V)*predict_cost # (K)
        action_index = np.argmax(weight)

        ########## Null action ##########
        if np.max(weight) < 0 and self.allow_null:
            self.logger.log_scalar(
                {
                    "train/predict_reward": 0,
                    "train/predict_cost": 0,
                },
                step=self.t,
            )
            return "None"
        ########## Null action ##########

        action = action_space[action_index]
        self.logger.log_scalar(
            {
                "train/predict_reward": predict_reward[action_index],
                "train/predict_cost": predict_cost[action_index],
            },
            step=self.t,
        )
        # print(action)
        if self.rmodel.concatenate:
            self.rinput_buffer.append(rmodel_input[action_index])
        else:
            self.rinput_buffer.append((rmodel_input_x[action_index], rmodel_input_a[action_index]))
        if self.cmodel.concatenate:
            self.cinput_buffer.append(cmodel_input[action_index])
        else:
            self.cinput_buffer.append((cmodel_input_x[action_index], cmodel_input_a[action_index]))
        return action
    
    def update(self, reward, cost):
        self.Q = max(self.Q + cost - self.b,0)

        if (len(self.r_buffer) == len(self.rinput_buffer)): # took null action in current round
            return
        
        self.r_buffer.append(reward)
        self.c_buffer.append(cost)
        self.logger.log_scalar(
                {
                    "train/Q": self.Q
                },
                step=self.t,
            )
        self.t += 1
        if len(self.rinput_buffer) > self.buffer_size:
            self.rmodel.online_update(np.array(self.rinput_buffer),np.array(self.r_buffer))
            self.cmodel.online_update(np.array(self.cinput_buffer),np.array(self.c_buffer))
            self.rinput_buffer = []
            self.cinput_buffer = []
            self.r_buffer = []
            self.c_buffer = []
            
