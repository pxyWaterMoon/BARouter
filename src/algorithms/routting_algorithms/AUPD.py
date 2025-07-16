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
        self.current_sample = None
        self.T = T


    def take_action(self, sample):
        self.current_sample = sample.copy()
        action_space = list(sample["available_models_description"].keys())
        sample_list = []
        for action_index, action in enumerate(action_space):
            sample = self.current_sample.copy()
            sample["model_name"] = action
            sample["model_description"] = sample["available_models_description"][action]
            sample["model_description_embedding"] = sample["available_models_description_embeddings"][action]
            sample["model_index"] = action_index
            sample_list.append(sample)
        predict_reward = self.rmodel.predict(sample_list)
        predict_cost = self.cmodel.predict(sample_list)


        # cmodel_input = self.embedding_fn(sample, concatenate=self.cmodel.concatenate)
        # # print(X.shape)
        # # X: (K, d)
        # if self.rmodel.concatenate:
        #     rmodel_input = self.embedding_fn(sample, concatenate=self.rmodel.concatenate)
        #     predict_reward:np.ndarray = self.rmodel.predict(rmodel_input) # (K)
        # else:
        #     rmodel_input_x, rmodel_input_a = self.embedding_fn(sample, concatenate=self.rmodel.concatenate)
        #     predict_reward:np.ndarray = self.rmodel.predict(rmodel_input_x, rmodel_input_a) # (K)
        # if self.cmodel.concatenate:
        #     cmodel_input = self.embedding_fn(sample, concatenate=self.cmodel.concatenate)
        #     predict_cost:np.ndarray = self.cmodel.predict(cmodel_input) # (K)
        # else:
        #     cmodel_input_x, cmodel_input_a = self.embedding_fn(sample, concatenate=self.cmodel.concatenate)
        #     predict_cost:np.ndarray = self.cmodel.predict(cmodel_input_x, cmodel_input_a) # (K)
        # print(predict_cost.shape)
        weight = predict_reward - (self.Q/self.V)*predict_cost # (K)
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

        action_index = np.argmax(weight)

        action = action_space[action_index]
        self.current_sample["model_index"] = action_index
        self.current_sample["model_name"] = action
        self.current_sample["model_description"] = sample["available_models_description"][action]
        self.current_sample["model_description_embedding"] = sample["available_models_description_embeddings"][action]
        self.current_sample["predict_reward"] = predict_reward[action_index]
        self.current_sample["predict_cost"] = predict_cost[action_index]
        self.logger.log_scalar(
            {
                "train/predict_reward": predict_reward[action_index],
                "train/predict_cost": predict_cost[action_index],
            },
            step=self.t,
        )
        return action
    
    def update(self, reward, cost, response):
        # self.r_buffer.append(reward)
        # self.c_buffer.append(cost)
        self.current_sample["reward"] = reward
        self.current_sample["cost"] = cost
        ret_sample = self.current_sample.copy()
        ret_sample["response"] = response
        self.rmodel.online_update(self.current_sample)
        self.cmodel.online_update(self.current_sample)
        self.current_sample = None
        self.Q = max(self.Q + cost - self.b,0)
        self.budget -= cost
        self.b = self.budget / max(self.T - self.t - 1, 1)
        self.logger.log_scalar(
                {
                    "train/Q": self.Q
                },
                step=self.t,
            )
        self.t += 1
        return ret_sample
