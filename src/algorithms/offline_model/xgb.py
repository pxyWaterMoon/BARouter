import pandas as pd
import numpy as np
from src.algorithms.offline_model.base_model import OfflineModel
from xgboost import XGBRegressor

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class XGB(OfflineModel):
    def __init__(self, depth=4):
        self.depth=depth
        self.model = XGBRegressor(max_depth=self.depth)

    def get_data(self, file_name:str, key:str):
        df = pd.read_parquet(file_name)
        prompt_embedding_list = df["prompt_embedding"].to_list()
        model_embedding_dict:list[dict] = df["available_models_description_embeddings"].to_list()
        gt:list[dict] = df["gt"].to_list()
        action_dict = df["available_models_description"][0]
        action_embedding_dict = df["available_models_description_embeddings"][0]
        self.action_list = list(action_dict.keys())
        self.action_embedding_list = [action_embedding_dict[action] for action in self.action_list]
        X = []
        y = []
        for i in range(len(prompt_embedding_list)):
            prompt_embedding = prompt_embedding_list[i]
            for model_name,model_embedding in model_embedding_dict[i].items():
                x = np.hstack([prompt_embedding,model_embedding])
                X.append(x)
                y.append(gt[i][model_name][key])
        return (X,y)
    
    def offline_training(self, dataset, key:str):
        X = []
        y = []
        for data in dataset:
            X.append(np.concatenate([data["prompt_embedding"],data["model_description_embedding"]]))
            y.append(data[key])
        X = np.array(X)
        y = np.array(y)
        print("xgb start training...")
        self.model.fit(X,y)

    def online_update(self, X, y):
        updated_model = XGBRegressor(max_depth=self.depth)
        print("xgb start retraining...")
        updated_model.fit(X, y, xgb_model=self.model)
        return updated_model
    
    def predict(self, X):
        res = self.model.predict(X)
        # print(res.shape)
        return res



if __name__=="__main__":
    model = XGB()
    model.predict(np.zeros((11,768+768)))
