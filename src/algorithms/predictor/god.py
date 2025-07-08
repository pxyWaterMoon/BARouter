import pandas as pd
import numpy as np
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class God(BasePredictor):
    def __init__(self, dataset, key):
        self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
        self.key = key
        self.concatenate = False
    
    def offline_training(self, dataset, key:str):
        pass

    def online_update(self, X, y):
        return
    
    def predict(self, prompts, actions):
        return np.array([self.gt[prompt][action][self.key] for prompt, action in zip(prompts, actions)])
    