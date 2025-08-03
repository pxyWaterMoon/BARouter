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

        ### debug ###
        random_prompt = list(self.gt.keys())[0]
        self.action_list = list(self.gt[random_prompt].keys())
        ### debug ###
    
    def offline_training(self, dataset, key:str):
        return

    def online_update(self, sample, global_step):
        return
    
    ### debug ###
    def get_prompt(self, sample_list:list[dict], key = None):
        
        first_prompt = sample_list[0]["prompt"]
        if any(sample["prompt"] != first_prompt for sample in sample_list):
            raise ValueError("Multiple prompts")
        
        return first_prompt
    ### debug ###    

    def predict(self, sample_list):
        prompt = self.get_prompt(sample_list)
        return np.array([self.gt[prompt][action][self.key] for action in self.action_list])

    # def predict(self, prompts, actions):
    #     return np.array([self.gt[prompt][action][self.key] for prompt, action in zip(prompts, actions)])
    