import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch

def split(data:list, rate) ->tuple[list,list]:
    n = int(len(data)*rate)
    return data[:n],data[n:]

class K_means_online(BasePredictor):
    def __init__(self, key, n_action, n_cluters=20):
        self.n_cluters=n_cluters
        self.X_list = []
        self.idx_list = []
        self.value_list = []

        self.values = np.zeros((n_cluters,n_action))
        self.counts = np.zeros((n_cluters,n_action))
        self.kmeans = None
        self.key=key
        self.cluster_time = 1

    def clustering(self, X, n_clutsers):
        self.kmeans = KMeans(
            n_clusters=n_clutsers,       # 指定聚类数量 K
            random_state=42,    # 随机种子（确保结果可复现）
            n_init=10           # 使用不同质心初始化的次数（默认=10）
        )
        self.kmeans.fit(X.astype(np.float64))
        cluster_labels = self.kmeans.labels_

        self.values[:] = 0
        self.counts[:] = 0

        for i,labels in enumerate(cluster_labels):
            self.values[labels,self.idx_list[i]] += self.value_list[i]
            self.counts[labels,self.idx_list[i]] += 1
        
        self.counts[self.counts == 0] = 1
        self.values /= self.counts

    def offline_training(self, dataset, key:str):
        return

    def online_update(self, sample, global_step):
        prompt_embedding = sample["prompt_embedding"]
        value = sample[self.key]
        action_idx = sample["model_index"]
        if len(self.X_list) < 2000*self.n_cluters: # kmeans
            self.X_list.append(prompt_embedding)
            self.idx_list.append(action_idx)
            self.value_list.append(value)
            if len(self.X_list) == self.cluster_time:
                k = 1 if len(self.X_list) < self.n_cluters*50 else self.n_cluters
                # print("global:",global_step)
                self.clustering(np.array(self.X_list),n_clutsers=k)
                self.cluster_time *= 2
                return
        label = self.kmeans.predict([prompt_embedding])[0]
        self.values[label,action_idx] = self.values[label][action_idx] * self.counts[label][action_idx] + value
        self.counts[label,action_idx] += 1
        self.values[label,action_idx] /= self.counts[label][action_idx]
        return
    
    def get_embedding(self, sample_list:list[dict], key = None):
        
        first_embedding = sample_list[0]["prompt_embedding"]
        if any(any(sample["prompt_embedding"] != first_embedding) for sample in sample_list):
            raise ValueError("Multiple prompts")
        
        return first_embedding

    def predict(self, sample_list):
        if self.kmeans is None:
            return np.ones(self.values.shape[1])
        embedding = self.get_embedding(sample_list)
        label = self.kmeans.predict([embedding])[0]
        return self.values[label]