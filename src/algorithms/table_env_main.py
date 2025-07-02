from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset, PromptOnlyDataLoader
from src.datasets.sftdata import SFTDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from src.metrics.table_base import TabelBasedModel
from src.algorithms.predictor.xgb import XGB
from src.algorithms.routting_algorithms.AUPD import AUPD
from src.logger import Logger
from tqdm import tqdm
from src.algorithms.utils import embedding_batch

def run():
    simuler_dataset = SimulerDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/train.parquet")
    SFT_dataset = SFTDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/test.parquet")

    dataset = PromptOnlyDataset(simuler_dataset)
    loader = PromptOnlyDataLoader(dataset)
    total_budget = 10 # dollar
    env_model = TabelBasedModel(simuler_dataset, budget=total_budget)
    logger = Logger(f"./outputs/logs/AUPD_budget_{total_budget}/")
    T = len(loader)
    rmodel = XGB()
    cmodel = XGB()
    rmodel.offline_training(SFT_dataset,key="reward")
    cmodel.offline_training(SFT_dataset,key="cost")
    agent = AUPD(rmodel,cmodel,logger,len(dataset),budget=total_budget/T)
    with tqdm(total=len(loader)) as pbar_total:
        for t, batch in enumerate(loader):
            X = embedding_batch(batch)
            rewards = []
            costs = []
            actions = []
            for sample, x in zip(batch, X):
                prompt = sample["prompt"]
                action_index = agent.take_action(x)
                if action_index == None:
                    response = None
                    reward = 0
                    cost = 0
                else:
                    action = list(sample["available_models_description"].keys())[action_index]
                    response, reward, cost = env_model.feedback(prompt, action)
                rewards.append(reward)
                costs.append(cost)
                actions.append(action)
                agent.update(x[action_index], reward, cost)
            current_reward = sum(rewards)/len(rewards)
            current_cost = sum(costs)/len(costs)
            logger.log_signal(actions, rewards, costs, t)
            logger.log_scalar(
                {
                    "train/current_reward": current_reward,
                    "train/curremt_cost": current_cost,
                    "train/average_reward": logger.get_log_value("rewards", range(t+1)),
                    "train/average_cost": logger.get_log_value("costs", range(t+1)),
                    "train/budget": env_model.budget
                },
                step=t,
            )
            pbar_total.update(1)
    logger.plot_action_log()
                

if __name__ == "__main__":
    run()


# def get_X(data):
#     X_list = []
#     # print(data["prompt_embedding"].shape)
#     for name,embedding in data["available_models_description_embeddings"].items():
#         # print(name,embedding.shape)
#         X_list.append(np.concatenate([data["prompt_embedding"],embedding]))
#     return X_list

# rmodel = XGB()
# cmodel = XGB()
# rmodel.offline_training(SFT_dataset,key="reward")
# cmodel.offline_training(SFT_dataset,key="cost")
# alg = AUPD(rmodel,cmodel,len(dataset),budget=2e-3)

# for t, data in enumerate(dataset):
#     X = get_X(data)
#     action = alg.take_action(X)
    
#     # print(len(X),X[0].shape)
#     x = data["prompt"]
#     avilable_models = list(data["available_models_description"].keys())
#     response, reward, cost = env_model.feedback(x, avilable_models[action])

#     # print(action, reward, cost)

#     alg.update(X[action],reward,cost)
#     # print(t)
#     logger.log_scalar(
#         {
#             "train/reward": reward,
#             "train/cost": cost,

#         },
#         step=t,
#     )
