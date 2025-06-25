from torch.utils.data import Dataset
from typing_extensions import TypedDict  # Python 3.10+
from typing_extensions import NotRequired  # Python 3.11+
import pandas as pd
from typing import Any

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




        