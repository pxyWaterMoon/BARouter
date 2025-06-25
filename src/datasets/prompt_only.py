from src.datasets.simulerdata import SimulerDataset
from torch.utils.data import Dataset
from typing_extensions import TypedDict  # Python 3.10+
from typing_extensions import NotRequired  # Python 3.11+
import pandas as pd
from typing import Any

class PromptOnlySample(TypedDict, total=False):
    prompt: NotRequired[str]
    prompt_embedding: NotRequired[list[float]]|None
    available_models_description: NotRequired[dict[str, str]]
    available_models_description_embeddings: NotRequired[dict[str, list[float]]]|None

class PromptOnlyDataset(Dataset):
    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1:
            # init with a SimulerDataset
            source_data = args[0]
            self.prompts = []
            self.prompts_embedding = []
            self.available_models_description = []
            self.available_models_description_embeddings = []
            for data in source_data:
                self.prompts.append(data["prompt"])
                self.prompts_embedding.append(data["prompt_embedding"])
                self.available_models_description.append(data["available_models_description"])
                self.available_models_description_embeddings.append(
                    data["available_models_description_embeddings"]
                )
            
        elif len(args) == 3:
            self.prompts = args[0]
            self.available_models_description = args[1]
            # self.embedding_path = args[2]
            self.prompts_embedding, self.available_models_description_embeddings = self.model_embed(args[2])
        
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index) -> Any:
        sample: PromptOnlySample = {
            "prompt": self.prompts[index],
            "prompt_embedding": self.prompts_embedding[index],
            "available_models_description": self.available_models_description[index],
            "available_models_description_embeddings": self.available_models_description_embeddings[index],
        }
        return sample


    def model_embed(self, embed_path):
        raise NotImplementedError