from torch.utils.data import Dataset
from typing_extensions import TypedDict  # Python 3.10+
from typing_extensions import NotRequired  # Python 3.11+
import pandas as pd
from typing import Any
import queue
import numpy as np

class SFTSample(TypedDict, total=False):
    prompt: NotRequired[str]
    prompt_embedding: NotRequired[list[float]]
    model_name: NotRequired[str]
    model_description: NotRequired[dict[str, str]]
    model_description_embedding: NotRequired[dict[str, list[float]]]
    reward: NotRequired[float]
    cost: NotRequired[float]

class SFTDataset(Dataset[SFTSample]):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.data = pd.read_parquet(file_path)
        self.K = len(self.data.iloc[0]["available_models_description"].keys())  # Number of available models
        self.prompt_num = len(self.data)
    
    def __len__(self):
        return self.prompt_num * self.K
    
    def __getitem__(self, index) -> Any:
        prompt_index = index // self.K
        model_index = index % self.K
        row = self.data.iloc[prompt_index]
        model_name = list(row["available_models_description"].keys())[model_index]
        sample: SFTSample = {
            "prompt": row.get("prompt"),
            "prompt_embedding": row.get("prompt_embedding"),
            "model_name": model_name,
            "model_description": row["available_models_description"][model_name],
            "model_description_embedding": row["available_models_description_embeddings"][model_name],
            "reward": row["gt"][model_name]["reward"] if "gt" in row and model_name in row["gt"] else None,
            "cost": row["gt"][model_name]["cost"] if "gt" in row and model_name in row["gt"] else None,
        }
        return sample

class SFTDataLoader:
    def __init__(self, dataset: SFTDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            import random
            random.shuffle(self.indices)
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            yield [self.dataset[idx] for idx in batch_indices]
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
class SFTBufferPool:
    def __init__(self, batch_size: int = 32, size: int = 128) -> None:
        self.dataset = []
        self.batch_size = batch_size
        self.max_size = size
        self.insert_pos = 0
    
    def add(self, sample: SFTSample):
        if len(self.dataset) < self.max_size:
            self.dataset.append(sample)
            self.insert_pos = (self.insert_pos + 1) % self.max_size
        else:
            self.dataset[self.insert_pos] = sample
            self.insert_pos = (self.insert_pos + 1) % self.max_size
    
    def get_batch(self):
        if len(self.dataset) < self.batch_size:
            raise ValueError
        random_indices = np.random.choice(len(self.dataset), self.batch_size, replace=False)
        return [self.dataset[idx] for idx in random_indices]

if __name__ == "__main__":
    # Example usage
    dataset = SFTDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/test.parquet")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}: {sample}")
        exit(0)