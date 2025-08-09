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
    def __init__(self, dataset, key, k=5):
        self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
        self.key = key
        self.K = k

        X = np.array([data["prompt_embedding"] for data in dataset])
        ### debug ###
        random_prompt = list(self.gt.keys())[0]
        action_list = list(self.gt[random_prompt].keys())
        y = np.array([
            [
                data["ground_truth"][action][key] for action in action_list
            ]for data in dataset
        ])
        # y.shape : (n_data,n_action)
        # self.scaler = StandardScaler()
        # scaled_X = self.scaler.fit_transform(X).astype(np.float64)
        self.kmeans = KMeans(
            n_clusters=k,       # 指定聚类数量 K
            random_state=42,    # 随机种子（确保结果可复现）
            n_init=10           # 使用不同质心初始化的次数（默认=10）
        )
        self.kmeans.fit(X.astype(np.float64))
        cluster_labels = self.kmeans.labels_

        self.mean_val = np.zeros((k, y.shape[1]))
        self.cluster_cnt = np.zeros((k,y.shape[1]))
        for cluster_id in range(k):
            # 获取当前聚类的所有样本索引
            cluster_indices = np.where(cluster_labels == cluster_id)
            self.cluster_cnt[cluster_id] = len(cluster_indices[0])*np.ones(y.shape[1])
            # 计算该聚类对应的y平均值
            self.mean_val[cluster_id] = np.mean(y[cluster_indices], axis=0)
        
        # print(self.mean_val)
        # print(self.cluster_cnt)
        # exit(0)
    
    def offline_training(self, dataset, key:str):
        return

    def online_update(self, sample, global_step):
        # print(sample.keys())
        # exit(0)
        prompt_embedding = sample["prompt_embedding"]
        action_idx = sample["model_index"]
        label = self.kmeans.predict([prompt_embedding])[0]
        value = sample[self.key]
        self.mean_val[label][action_idx] = self.mean_val[label][action_idx] * self.cluster_cnt[label][action_idx] + value
        self.cluster_cnt[label][action_idx] += 1
        self.mean_val[label][action_idx] /= self.cluster_cnt[label][action_idx]
        return
    
    def get_embedding(self, sample_list:list[dict], key = None):
        
        first_embedding = sample_list[0]["prompt_embedding"]
        if any(any(sample["prompt_embedding"] != first_embedding) for sample in sample_list):
            raise ValueError("Multiple prompts")
        
        return first_embedding

    def predict(self, sample_list):
        embedding = self.get_embedding(sample_list)
        # return np.array([self.gt[prompt][action][self.key] for action in self.action_list])
        # embedding = self.scaler.transform([embedding])
        # print(embedding.dtype)
        label = self.kmeans.predict([embedding])[0]
        ret = self.mean_val[label]
        return ret