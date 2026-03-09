import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch


class Gradient(BasePredictor):
    def __init__(self, dataset, budget_perround, k=5):

        X = np.array([data["prompt_embedding"] for data in dataset])
        X = X[:200]
        self.N = X.shape[0]

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
        labels = self.kmeans(k, X)
        rkj = np.zeros((k,len(self.action_list)))
        ckj = np.zeros((k,len(self.action_list)))
        count_labels = np.zeros(k)
        for i, label in enumerate(labels):
            rkj[label] += r_array[i]
            ckj[label] += c_array[i]
            count_labels[label] += 1
        self.policy = self.get_policy(rkj,ckj,budget_perround*X.shape[0])
    
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
        
    def get_policy(self, matrix_r:np.ndarray, matrix_c:np.ndarray, total_budget):
        K, M = matrix_r.shape

        c_obj = -matrix_r.flatten()

        A_ub = matrix_c.reshape(1,-1)
        b_ub = np.array([total_budget])

        A_eq = np.zeros((K,K*M))
        for i in range(K):
            A_eq[i,i*M:(i+1)*M] = 1
        b_eq = np.ones(K)
        
        bounds = [(0, None)] * (K*M)

        from scipy.optimize import linprog
        result = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not result.success:
            print("Not feassiable")
            return np.ones((K,M)) / M
        print("Average reward in offline data:",result.fun/self.N)
        return result.x.reshape((K,M))


    # def gradient_decent(self,r,c,buget,max_iter=int(1e3),eta=1e-5):

    #     def get_x(current_lambda):
    #         idx = np.argmin(current_lambda*c - r, axis=1)
    #         x = np.zeros(r.shape)
    #         # 让x每一行的idx列为1
    #         x[np.arange(r.shape[0]), idx] = 1
    #         return x
        
    #     def iter_lambda(current_lambda):
    #         return max(0, current_lambda + eta*(np.sum(c*x) - buget))

    #     lambda_list = [0]
    #     lambda_ = 0
    #     for _ in range(max_iter):
    #         x = get_x(lambda_)
    #         lambda_ = iter_lambda(lambda_)
    #         if len(lambda_list)>0 and abs(lambda_list[-1] - lambda_) < 1e-5 :
    #             break
    #         lambda_list.append(lambda_)
    #     # print(lambda_list)
    #     return x, lambda_

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