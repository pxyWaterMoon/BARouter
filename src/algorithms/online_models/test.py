from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset
from src.datasets.sftdata import SFTDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from src.metrics.table_base import TabelBasedModel
from src.algorithms.offline_model.xgb import XGB
from src.algorithms.online_models.AUPD import AUPD
from src.logger import Logger

simuler_dataset = SimulerDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/train.parquet")
SFT_dataset = SFTDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/test.parquet")

dataset = PromptOnlyDataset(simuler_dataset)
env_model = TabelBasedModel(simuler_dataset)
logger = Logger("./outputs/logs/AUPD_2e-3/")

def get_X(data):
    X_list = []
    # print(data["prompt_embedding"].shape)
    for name,embedding in data["available_models_description_embeddings"].items():
        # print(name,embedding.shape)
        X_list.append(np.concatenate([data["prompt_embedding"],embedding]))
    return X_list

rmodel = XGB()
cmodel = XGB()
rmodel.offline_training(SFT_dataset,key="reward")
cmodel.offline_training(SFT_dataset,key="cost")
alg = AUPD(rmodel,cmodel,len(dataset),budget=2e-3)

for t, data in enumerate(dataset):
    X = get_X(data)
    action = alg.take_action(X)
    
    # print(len(X),X[0].shape)
    x = data["prompt"]
    avilable_models = list(data["available_models_description"].keys())
    response, reward, cost = env_model.feedback(x, avilable_models[action])

    # print(action, reward, cost)

    alg.update(X[action],reward,cost)
    # print(t)
    logger.log_scalar(
        {
            "train/reward": reward,
            "train/cost": cost,

        },
        step=t,
    )

