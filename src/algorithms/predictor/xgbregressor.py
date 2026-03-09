import pandas as pd
import numpy as np
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch
from xgboost import XGBRegressor

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class XGBRegressorPredictor(BasePredictor):
    def __init__(self, key, SFT_dataset=None, buffer_size=256, offline = False):
        self.model = XGBRegressor(max_depth=4,learning_rate=0.01)
        self.key = key
        self.offline = offline
        self.buffer = []
        self.buffer_size = buffer_size
        self.X = []
        self.y = []
        if SFT_dataset is not None:
            for data in SFT_dataset:
                self.X.append(np.concatenate([data["prompt_embedding"],data["model_description_embedding"]]))
                self.y.append(data[key])
            if not self.offline:
                self.X = self.X[:200]
                self.y = self.y[:200]
                print("online!")
            self.X = np.array(self.X)
            self.y = np.array(self.y)
        self.model.fit(self.X,self.y)
        print(f"Successfully trained the predictor of {key}.")
    
    def sample2input(self, sample_list:list[dict], key = None):
        if all("prompt_embedding" not in sample for sample in sample_list) or all("model_description_embedding" not in sample for sample in sample_list):
            raise ValueError("The input sample_list must contain 'prompt_embedding' and 'model_description_embedding' keys for XGBoost predictor.")
        prompt_embeddings = [sample["prompt_embedding"] for sample in sample_list]
        model_description_embeddings = [sample["model_description_embedding"] for sample in sample_list]
        X = np.concatenate([np.array(prompt_embeddings),np.array(model_description_embeddings)],axis=1)
        if key is not None:
            y = [sample[key] for sample in sample_list]
            return X, np.array(y)
        return X

    def offline_training(self):
        return

    def online_update(self, sample, global_step):
        if self.offline:
            return
        return
        self.buffer.append(sample)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.append(sample)
            new_X, new_y = self.sample2input(self.buffer, key=self.key)
            self.X = np.concatenate([self.X,new_X],axis=0)
            self.y = np.concatenate([self.y,new_y],axis=0)
            self.buffer = []
            self.model.fit(self.X,self.y)
    
    def predict(self, sample_list):
        X = self.sample2input(sample_list)
        res = self.model.predict(X)
        # print(res.shape)
        return res



if __name__=="__main__":
    model = XGBRegressorPredictor()
    model.predict(np.zeros((11,768+768)))
