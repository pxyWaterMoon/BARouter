import pandas as pd
import numpy as np
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class God(BasePredictor):
    def __init__(self, dataset):
        self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
    
    def offline_training(self, dataset, key:str):
        self.key = key

    def online_update(self, X, y):
        return
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = X.tolist()
        res = []
        for x in X:
            prompt = x[0]
            if prompt not in self.gt:
                raise ValueError(f"Prompt {prompt} not found in ground truth.")
            res.append(self.gt[prompt][self.key])
        return np.array(res)



if __name__=="__main__":
    model = XGB()
    model.predict(np.zeros((11,768+768)))
