import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch


class Gradient(BasePredictor):
    def __init__(self, dataset, budget_perround, k=5):

        X = np.array([data["prompt_embedding"] for data in dataset])

        print(X.shape)

        self.gt = [data["ground_truth"] for data in dataset]
        self.K = k

        r_array = []
        c_array = []
        self.action_list = list(dataset[0]["ground_truth"].keys())
        for data in self.gt:
            r_array.append([data[llm]["reward"] for llm in self.action_list])
            c_array.append([data[llm]["cost"] for llm in self.action_list])
        
        r_array = np.array(r_array)
        c_array = np.array(c_array)
        print(r_array.shape,c_array.shape)

        budget = budget_perround * len(dataset)

        opt_action, opt_lambda = self.gradient_decent(r_array,c_array,budget)
        print(opt_action.shape,opt_lambda)
        opt_action = np.argmax(opt_action,axis=1)
        print(opt_action.shape)
        labels = self.kmeans(k, X)
        print(labels.shape)

        # sum = 0
        # for i, action in enumerate(opt_action):
        #     sum += r_array[i][action]
        # print(sum/len(opt_action))
        # exit(0)

        self.policy = np.zeros((k,len(self.action_list)))
        # 遍历每个数据点的下标和标签
        for idx, label in enumerate(labels):
            self.policy[label,opt_action[idx]] += 1
        
        # 1. 计算每一行的和
        row_sums = self.policy.sum(axis=1, keepdims=True)  # keepdims=True 保持二维结构

        # 2. 处理零和的行（避免除以零）
        #    将零和替换为1，这样0/1=0，保持原值
        row_sums[row_sums == 0] = 1

        # 3. 归一化：每行除以该行的和
        self.policy /= row_sums
        
        print(np.max(self.policy))
        
    
    def kmeans(self, n_clusters, X):
        self.kmeans = KMeans(
            n_clusters=n_clusters,       # 指定聚类数量 K
            random_state=42,    # 随机种子（确保结果可复现）
            n_init=10           # 使用不同质心初始化的次数（默认=10）
        )
        self.kmeans.fit(X.astype(np.float64))
        cluster_labels = self.kmeans.labels_
        # 根据聚类结果将下标分为 n_clusters 类

        return cluster_labels
        


    def gradient_decent(self,r,c,buget,max_iter=int(1e3),eta=1e-5):

        def get_x(current_lambda):
            idx = np.argmin(current_lambda*c - r, axis=1)
            x = np.zeros(r.shape)
            # 让x每一行的idx列为1
            x[np.arange(r.shape[0]), idx] = 1
            return x
        
        def iter_lambda(current_lambda):
            return max(0, current_lambda + eta*(np.sum(c*x) - buget))

        lambda_list = [0]
        lambda_ = 0
        for _ in range(max_iter):
            x = get_x(lambda_)
            lambda_ = iter_lambda(lambda_)
            if len(lambda_list)>0 and abs(lambda_list[-1] - lambda_) < 1e-5 :
                break
            lambda_list.append(lambda_)
        # print(lambda_list)
        return x, lambda_

    def take_action(self, sample):
        self.current_sample = sample.copy()
        # print(sample.keys())
        # exit(0)
        prompt_embedding = sample["prompt_embedding"]
        
        label = self.kmeans.predict([prompt_embedding])[0]
        action_index = np.random.choice(len(self.policy[label]),p=self.policy[label])
        action = self.action_list[action_index]
        self.current_sample["model_name"] = action
        self.current_sample["model_index"] = action_index
        return action
        
    
    def update(self, reward, cost, response):
        self.current_sample["reward"] = reward
        self.current_sample["cost"] = cost
        ret_sample = self.current_sample.copy()
        ret_sample["response"] = response
        self.current_sample = None
        return ret_sample


if __name__ == "__main__":
    from src.datasets.simulerdata import SimulerDataset
    path = "data/processed/bert-base-uncased/routerbench_0shot_gpt4_llama270b/online_cost_scale_1000_cleaned.parquet"
    path_sp = "data/processed/bert-base-uncased/sprout/offline_cost_scale_1000.parquet"
    dataset = SimulerDataset(file_path=path_sp)
    gra = Gradient(dataset, 0.5, key="reward",k=5)