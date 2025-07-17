from src.algorithms.routting_algorithms.base_model import OnlineModel
from src.algorithms.predictor.xgb import XGB
from src.logger import Logger
import numpy as np

class FixAction(OnlineModel):
    def __init__(self, action, T, logger):
        self.action=action
        self.t = 0
        self.T = T
        self.logger = logger
        self.current_sample = None

    def take_action(self, sample):
        self.current_sample = sample.copy()
        action_space = list(sample["available_models_description"].keys())
        if self.action not in action_space:
            raise ValueError(f"Action {self.action} not in action space {action_space}")
        action_index = action_space.index(self.action)
        self.current_sample["model_index"] = action_index
        self.current_sample["model_name"] = self.action
        self.current_sample["model_description"] = sample["available_models_description"][self.action]
        self.current_sample["model_description_embedding"] = sample["available_models_description_embeddings"][self.action]
        return self.action
    
    def update(self, reward, cost, response):
        self.current_sample["reward"] = reward
        self.current_sample["cost"] = cost
        ret_sample = self.current_sample.copy()
        ret_sample["response"] = response
        return ret_sample
