from src.algorithms.routting_algorithms.base_model import OnlineModel
from src.algorithms.predictor.base_model import BasePredictor
from src.logger import Logger
import numpy as np

class LOE2D(OnlineModel):
    def __init__(self, rmodel:BasePredictor, cmodel:BasePredictor, logger:Logger, T, budget, K, U, embedding_fn, buffer_size = 1024):
        self.budget = budget
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.b = self.budget / T

        self.K = K
        self.Q = 0
        self.V = self.b * np.sqrt(T * U * np.log(T))
        # self.V = self.b * np.sqrt(T)
        self.beta = 1
        self.gamma = K * np.sqrt(T/U)
        self.buffer_size = buffer_size
        self.t = 0
        self.logger = logger
        self.embedding_fn = embedding_fn
        self.current_sample = None
        self.T = T

    def inversegap(self, scorelist, gamma, K):
        optimal_index = np.argmax(scorelist)
        Lagrangian_gap = scorelist[optimal_index] - scorelist
        # print(f"Lagrangian_gap: {Lagrangian_gap}")
        eps = 1e-4
        left, right = 1, K
        while(True):
            mid = (left + right)/2
            s = np.sum(1 / (mid + 2 * gamma * Lagrangian_gap))
            if np.abs(s-1) < eps:
                break
            if s > 1:
                left = mid
            else:
                right = mid
        pi = 1 / (mid + 2 * gamma * Lagrangian_gap)
        pi = pi.astype(np.float64) / np.sum(pi)
        # print(f"pi: {pi}")
        return np.random.choice(len(pi),p=pi)

    def take_action(self, sample):
        self.current_sample = sample.copy()
        action_space = list(sample["available_models_description"].keys())
        sample_list = []
        for action_index, action in enumerate(action_space):
            sample = self.current_sample.copy()
            sample["model_name"] = action
            sample["model_description"] = sample["available_models_description"][action]
            sample["model_description_embedding"] = sample["available_models_description_embeddings"][action]
            sample["model_index"] = action_index
            sample_list.append(sample)
        predict_reward = self.rmodel.predict(sample_list)
        predict_cost = self.cmodel.predict(sample_list)

        # predict_cost -= self.b

        # weight = predict_reward - (self.Q/self.V)*predict_cost # (K)
        # self.current_sample["weight"] = weight
        # self.current_sample["all_predict_reward"] = predict_reward
        # self.current_sample["all_predict_cost"] = predict_cost
        # ########## Null action ##########
        # if np.max(weight) < 0 and self.allow_null:
        #     self.logger.log_scalar(
        #         {
        #             "train/predict_reward": 0,
        #             "train/predict_cost": 0,
        #         },
        #         step=self.t,
        #     )
        #     return "None"
        # ########## Null action ##########
        # weight *= self.eta*np.sqrt(self.t+1)
        # delta = np.max(weight) - 10
        # weight -= delta
        # # print(weight)
        # weight = np.exp(weight)
        # weight = weight / np.sum(weight)
        weight = predict_reward - (self.Q/self.V) * predict_cost
        # print(self.Q,self.V,weight)
        action_index = self.inversegap(weight, self.gamma * self.beta, self.K)

        action = action_space[action_index]
        self.current_sample["model_index"] = action_index
        self.current_sample["model_name"] = action
        self.current_sample["model_description"] = sample["available_models_description"][action]
        self.current_sample["model_description_embedding"] = sample["available_models_description_embeddings"][action]
        self.current_sample["predict_reward"] = predict_reward[action_index]
        self.current_sample["predict_cost"] = predict_cost[action_index]
        
        self.logger.log_scalar(
            {
                "train/predict_reward": predict_reward[action_index],
                "train/predict_cost": predict_cost[action_index],
            },
            step=self.t,
        )
        return action
    
    def update(self, reward, cost, response):
        # self.r_buffer.append(reward)
        # self.c_buffer.append(cost)
        self.current_sample["reward"] = reward
        self.current_sample["cost"] = cost
        ret_sample = self.current_sample.copy()
        ret_sample["response"] = response
        self.rmodel.online_update(self.current_sample, self.t)
        self.cmodel.online_update(self.current_sample, self.t)
        self.current_sample = None
        self.logger.log_scalar(
                {
                    "train/Q": self.Q
                },
                step=self.t,
            )
        self.t += 1

        self.Q = max(self.Q + cost - self.b, 0)
        self.beta = self.V / (self.V + self.Q)
        return ret_sample


# class LOE:
#     def __init__(self, rmodel = None, cmodel = None, K=4, T=5000, U=30, b = 1):
#         self.K = K
#         # LOE2D
#         self.Q = 0
#         self.beta = 1
#         self.V = np.sqrt(T * U * np.log(T))
#         self.gamma = self.K * np.sqrt(T / U)
#         self.b = b
#         self.rmodel = rmodel
#         self.cmodel = cmodel

#     def inversegap(self, scorelist, gamma, K):
#         optimal_index = np.argmax(scorelist)
#         Lagrangian_gap = scorelist[optimal_index] - scorelist
#         eps = 1e-4
#         left, right = 1, K
#         while(True):
#             mid = (left + right)/2
#             s = np.sum(1 / (mid + gamma * Lagrangian_gap))
#             if np.abs(s-1) < eps:
#                 break
#             if s > 1:
#                 left = mid
#             else:
#                 right = mid
#         pi = 1 / (left + gamma * Lagrangian_gap)
#         pi = pi.astype(np.float64) / np.sum(pi)
#         return np.argmax(np.random.multinomial(1,pi))
    

#     def take_action(self, context):
#         #model predict
#         preward = self.rmodel.predict(context)#.reshape(self.K, 1)
#         pcost = self.cmodel.predict(context)#.reshape(self.K, 1)
#         # LOE2D decision
#         Lagweight = preward - (self.Q / self.V) * pcost
#         return self.inversegap(Lagweight, self.gamma * self.beta, self.K)


#     def update(self, context, action, reward, cost):
#         # model update
#         self.rmodel.update(context[action], reward)
#         self.cmodel.update(action, cost)
#         self.Q = self.Q + cost
#         if self.Q < 0:
#             self.Q = 0
#         self.beta = self.V / (self.V + self.Q)