from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset
from torch.utils.data import DataLoader
import numpy as np
from src.metrics.table_base import TabelBasedModel

simuler_dataset = SimulerDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/train.parquet")

dataset = PromptOnlyDataset(simuler_dataset)
env_model = TabelBasedModel(simuler_dataset)

for t, data in enumerate(dataset):
    print(f"========== step {t} ===============")
    x = data["prompt"]
    print(f"PROMPT: {x}")
    avilable_models = list(data["available_models_description"].keys())

    a = np.random.choice(avilable_models)
    print("CHOOSED LM:", a)
    response, reward, cost = env_model.feedback(x, a)
    print(f"RESPONSE: {response}")
    print(f"REWARD: {reward}")
    print(f"COST: {cost}")
    if t > 10:
        break  # Remove this line to process all batches


