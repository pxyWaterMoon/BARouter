from torch.utils.data import Dataset
from typing_extensions import TypedDict  # Python 3.10+
from typing_extensions import NotRequired  # Python 3.11+
import pandas as pd
from typing import Any
import random
class SimulerSample(TypedDict, total=False):
    prompt: NotRequired[str]
    prompt_embedding: NotRequired[list[float]]
    available_models_description: NotRequired[dict[str, str]]
    available_models_description_embeddings: NotRequired[dict[str, list[float]]]
    ground_truth: NotRequired[dict[str, dict[str, float | str]]]

class SimulerDataset(Dataset[SimulerSample]):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.data = pd.read_parquet(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Any:
        row = self.data.iloc[index]
        sample: SimulerSample = {
            "prompt": row.get("prompt"),
            "prompt_embedding": row.get("prompt_embedding"),
            "available_models_description": row.get("available_models_description"),
            "available_models_description_embeddings": row.get("available_models_description_embeddings"),
            "ground_truth": row.get("gt"),
        }
        return sample

class SimulerDataLoader:
    def __init__(self, dataset: SimulerDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        self.current_index = 0
        
    def __len__(self):
        return len(self.indices)
    
    def get_sample(self):
        if self.current_index >= len(self.indices):
            raise StopIteration("No more samples available.")
        index = self.indices[self.current_index]
        sample = self.dataset[index]
        self.current_index += 1
        return sample

    def reset(self):
        self.current_index = 0

    def get_action_space(self):
        if len(self.dataset) == 0:
            return []
        first_sample = self.dataset[0]
        if "available_models_description" in first_sample:
            return list(first_sample["available_models_description"].keys())
        return []

class RandomSimulerDataLoader:
    def __init__(self, dataset: SimulerDataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        self.length = len(dataset)

        
    def __len__(self):
        return 99999999999

    def get_sample(self):
        index = random.randint(0, self.length - 1)
        sample = self.dataset[index]
        return sample

    def reset(self):
        pass

    def get_action_space(self):
        if len(self.dataset) == 0:
            return []
        first_sample = self.dataset[0]
        if "available_models_description" in first_sample:
            return list(first_sample["available_models_description"].keys())
        return []

        