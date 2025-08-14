import pandas as pd
import numpy as np
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch
from xgboost import XGBRegressor

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class XGB(BasePredictor):
    def __init__(self, key, SFT_dataset=None, buffer_size=128, offline = False):
        self.model = XGBRegressor(max_depth=4,learning_rate=0.01)
        self.key = key
        self.offline = offline
        self.buffer = []
        self.buffer_size = buffer_size
        if SFT_dataset is not None:
            self.offline_training(SFT_dataset, key=key)
    
    # def get_data(self, file_name:str, key:str):
    #     df = pd.read_parquet(file_name)
    #     prompt_embedding_list = df["prompt_embedding"].to_list()
    #     model_embedding_dict:list[dict] = df["available_models_description_embeddings"].to_list()
    #     gt:list[dict] = df["gt"].to_list()
    #     action_dict = df["available_models_description"][0]
    #     action_embedding_dict = df["available_models_description_embeddings"][0]
    #     self.action_list = list(action_dict.keys())
    #     self.action_embedding_list = [action_embedding_dict[action] for action in self.action_list]
    #     X = []
    #     y = []
    #     for i in range(len(prompt_embedding_list)):
    #         prompt_embedding = prompt_embedding_list[i]
    #         for model_name,model_embedding in model_embedding_dict[i].items():
    #             x = np.hstack([prompt_embedding,model_embedding])
    #             X.append(x)
    #             y.append(gt[i][model_name][key])
    #     return (X,y)
    
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

    def offline_training(self, dataset, key:str):
        X = []
        y = []
        for data in dataset:
            X.append(np.concatenate([data["prompt_embedding"],data["model_description_embedding"]]))
            y.append(data[key])
        # X, y = embedding_batch(dataset, key=key)
        X = np.array(X)
        y = np.array(y)
        if not self.offline:
            X = X[:10]
            y = y[:10]
            print("online!")
        self.model.fit(X,y)
        print(f"Successfully trained the predictor of {key}.")

    def online_update(self, sample, global_step):
        self.buffer.append(sample)
        if len(self.buffer) >= self.buffer_size:
            self.buffer.append(sample)
            X, y = self.sample2input(self.buffer, key=self.key)
            # print(X.shape, y.shape)
            updated_model = XGBRegressor(max_depth=4)
            # print("xgb start retraining...")
            updated_model.fit(X, y, xgb_model=self.model)
            self.model = updated_model
            self.buffer = []
        # return updated_model
    
    def predict(self, sample_list):
        X = self.sample2input(sample_list)
        res = self.model.predict(X)
        # print(res.shape)
        return res



if __name__=="__main__":
    model = XGB()
    model.predict(np.zeros((11,768+768)))
