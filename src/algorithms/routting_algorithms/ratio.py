from src.algorithms.routting_algorithms.base_model import OnlineModel
from src.algorithms.predictor.base_model import BasePredictor
from src.logger import Logger
import numpy as np

class Ratio(OnlineModel):
    def __init__(self, rmodel:BasePredictor, cmodel:BasePredictor, logger:Logger, T, budget, embedding_fn):
        self.budget = budget
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.t = 0
        self.logger = logger
        self.embedding_fn = embedding_fn

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

        predict_reward = np.clip(predict_reward, 0, 1)
        predict_cost = np.clip(predict_reward, 1e-6, None)

        weight = predict_reward / predict_cost # (K)
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
        self.current_sample["reward"] = reward
        self.current_sample["cost"] = cost
        ret_sample = self.current_sample.copy()
        ret_sample["response"] = response
        self.current_sample = None
        self.budget -= cost
        self.t += 1
        return ret_sample
