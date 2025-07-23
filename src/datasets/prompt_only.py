from src.datasets.simulerdata import SimulerDataset
from torch.utils.data import Dataset
from typing_extensions import TypedDict  # Python 3.10+
from typing_extensions import NotRequired  # Python 3.11+
import pandas as pd
from typing import Any

class PromptOnlySample(TypedDict, total=False):
    prompt: NotRequired[str]
    prompt_embedding: NotRequired[list[float]]|None
    available_models_description: NotRequired[dict[str, str]]|None
    available_models_description_embeddings: NotRequired[dict[str, list[float]]]|None

class PromptOnlyDataset(Dataset):
    def __init__(self, file_path) -> None:
        self.data = pd.read_parquet(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        row = self.data.iloc[index]
        sample = row["question"]
        gt = row["answer"] if "answer" in row.keys() else None
        return sample, gt          

# class PromptOnlyDataLoader:
#     def __init__(self, dataset: PromptOnlyDataset, batch_size: int = 1, shuffle: bool = True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = list(range(len(dataset)))
#         if self.shuffle:
#             import random
#             random.shuffle(self.indices)
    
#     def __iter__(self):
#         for i in range(0, len(self.dataset), self.batch_size):
#             batch_indices = self.indices[i:i + self.batch_size]
#             yield [self.dataset[idx] for idx in batch_indices]
    
#     def __len__(self):
#         return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
class PromptOnlyDataLoader:
    def __init__(self, dataset: PromptOnlyDataset, shuffle: bool = True, embed_fn=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        self.embed_fn = embed_fn
        self.current_index = 0
        
    def __len__(self):
        return len(self.dataset)
    
    def get_sample(self):
        if self.current_index >= len(self.indices):
            raise StopIteration("No more samples available.")
        index = self.indices[self.current_index]
        sample, gt = self.dataset[index]
        sample_embedding = self.embed_fn(sample) if self.embed_fn else None
        self.current_index += 1
        return {
            "prompt": sample,
            "prompt_embedding": sample_embedding,
            "available_models_description": None,
            "available_models_description_embeddings": None,
            "gt": gt
        }
    
    def reset(self):
        self.current_index = 0
        
