import pandas as pd
import numpy as np
# from sklearn.neighbors import KNeighborsRegressor
from cuml.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class KNNPredictor:
    def __init__(self, X, y, k):
        self.k = k
        self.X = None
        self.y = None
        self.update(X, y)
    
    def update(self, X, y):
        
        self.X = X if self.X is None else np.vstack([self.X, X])
        # if self.y is not None:
        #     print(self.y.shape,y.shape)
        self.y = y if self.y is None else np.concatenate([self.y, np.array(y)])
        X = np.array(self.X).astype(np.float64)
        y = np.array(self.y).astype(np.float64)
        self.knn = KNeighborsRegressor(n_neighbors=self.k,
                                    weights='uniform',
                                    algorithm='auto',
                                    p=2,
                                )
        self.knn.fit(X, y)

    def predict(self, x):
        return self.knn.predict(x)[0]    

class OLKNN(BasePredictor):
    def __init__(self, dataset, key, k=5, offline = True):
        self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
        self.key = key
        self.K = k

        # if not offline:
        #     dataset=dataset[:10]

        X = np.array([data["prompt_embedding"] for data in dataset])
        ### debug ###
        random_prompt = list(self.gt.keys())[0]
        self.action_list = list(self.gt[random_prompt].keys())

        NN = self.K

        if offline:
            self.knn_dict = {action: KNNPredictor(X, np.array([data["ground_truth"][action][key] for data in dataset]), k) for action in self.action_list}
        else:
            self.knn_dict = {action: KNNPredictor(X[:NN], np.array([data["ground_truth"][action][key] for data in dataset])[:NN], k) for action in self.action_list}
        self.buffer_maxsize = 64
        self.clear_buffer()

        # y = np.array([
        #     [
        #         data["ground_truth"][action][key] for action in action_list
        #     ]for data in dataset
        # ])
        # self.knn = KNeighborsRegressor(n_neighbors=k,
        #                                weights='distance',
        #                                algorithm='auto',
        #                                p=2,
        #                             )
        # self.knn.fit(X.astype(np.float64), y.astype(np.float64))
        # y.shape : (n_data,n_action)
        # self.scaler = StandardScaler()
        # scaled_X = self.scaler.fit_transform(X).astype(np.float64)
        # self.kmeans = KMeans(
        #     n_clusters=k,       # 指定聚类数量 K
        #     random_state=42,    # 随机种子（确保结果可复现）
        #     n_init=10           # 使用不同质心初始化的次数（默认=10）
        # )
        # self.kmeans.fit(X.astype(np.float64))
        # cluster_labels = self.kmeans.labels_

        # self.mean_val = np.zeros((k, y.shape[1]))
        # for cluster_id in range(k):
        #     # 获取当前聚类的所有样本索引
        #     cluster_indices = np.where(cluster_labels == cluster_id)
        #     # 计算该聚类对应的y平均值
        #     self.mean_val[cluster_id] = np.mean(y[cluster_indices], axis=0)
        
        # print(self.mean_val)

    def clear_buffer(self):
        self.buffer = {action: ([], []) for action in self.action_list}
        self.buffer_cursize = 0
        
    
    def offline_training(self, dataset, key:str):
        return

    def online_update(self, sample, global_step):
        action = sample["model_name"]
        prompt_embedding = sample["prompt_embedding"]
        value = sample[self.key]
        self.buffer[action][0].append(prompt_embedding)
        self.buffer[action][1].append(value)
        self.buffer_cursize += 1
        if self.buffer_cursize >= self.buffer_maxsize:
            for action in self.action_list:
                if len(self.buffer[action][0]) > 0:
                    X, y = self.buffer[action]
                    self.knn_dict[action].update(X, y)
            self.clear_buffer()
    
    def get_embedding(self, sample_list:list[dict], key = None):
        
        first_embedding = sample_list[0]["prompt_embedding"]
        if any(any(sample["prompt_embedding"] != first_embedding) for sample in sample_list):
            raise ValueError("Multiple prompts")
        
        return first_embedding

    def predict(self, sample_list):
        embedding = self.get_embedding(sample_list)
        # return np.array([self.gt[prompt][action][self.key] for action in self.action_list])
        # embedding = self.scaler.transform([embedding])
        embedding = embedding.reshape(1,-1).astype(np.float64)
        # x = self.knn.predict(embedding)
        # print(f"[DEBUG] x: {x}")
        # return x[0]
        return np.array([self.knn_dict[action].predict(embedding) for action in self.action_list])
        # return self.knn.predict(embedding)[0]
    
