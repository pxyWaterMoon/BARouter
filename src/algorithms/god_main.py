from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset, PromptOnlyDataLoader
from src.datasets.sftdata import SFTDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from src.metrics.table_base import TabelBasedModel
from src.algorithms.predictor.god import God
from src.algorithms.routting_algorithms.AUPD import AUPD
from src.logger import Logger
from tqdm import tqdm
from src.algorithms.utils import embedding_batch

def run():
    simuler_dataset = SimulerDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/train.parquet")
    SFT_dataset = SFTDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/test.parquet")

    dataset = PromptOnlyDataset(simuler_dataset)
    loader = PromptOnlyDataLoader(dataset)
    total_budget = 50 # dollar
    env_model = TabelBasedModel(simuler_dataset, budget=total_budget)
    logger = Logger(f"./outputs/logs/god_AUPD_budget_{total_budget}/")
    T = len(loader)
    rmodel = God(simuler_dataset, key="reward")
    cmodel = God(simuler_dataset, key="total_cost")
    pre_budget = total_budget / T
    V = pre_budget * np.sqrt(T)
    Q = 0
    # agent = AUPD(rmodel,cmodel,logger,len(dataset),budget=total_budget/T)
    with tqdm(total=len(loader)) as pbar_total:
        for t, batch in enumerate(loader):
            X = embedding_batch(batch)
            prompts = []
            rewards = []
            costs = []
            actions = []
            for sample in batch:
                prompt = sample["prompt"]
                prompts.append(prompt)
                available_models = list(sample["available_models_description"].keys())
                predict_reward = []
                predict_cost = []
                for model in available_models:
                    predict_reward.append(rmodel.predict(prompt, model))
                    predict_cost.append(cmodel.predict(prompt, model))
                predict_reward = np.array(predict_reward)
                predict_cost = np.array(predict_cost)
                weight = predict_reward - (Q / V) * predict_cost
                action_index = np.argmax(weight)
                Q = max(Q + predict_cost[action_index] - pre_budget, 0)
                logger.log_scalar(
                    {
                        "train/Q": Q
                    },
                    step=t,
                )
                action = available_models[action_index]
                response, reward, cost = env_model.feedback(prompt, action)
                rewards.append(reward)
                costs.append(cost)
                actions.append(action)

            current_reward = sum(rewards)/len(rewards)
            current_cost = sum(costs)/len(costs)
            logger.log_signal(prompts, actions, rewards, costs, t)
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
    logger.save_history()
                

if __name__ == "__main__":
    run()
