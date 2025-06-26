import torch
import numpy as np

def extend_prompt(batch: list):
    bs = len(batch)
    

def embedding_batch(batch: list):
    bs = len(batch)
    sample_keys = batch[0].keys()
    prompt_embeddings = []
    model_description_embeddings = []

    if "available_models_description_embeddings" in sample_keys: # online data
        for index in range(bs):
            prompt_embeddings.append(batch[index]["prompt_embedding"])
            model_description_embeddings.append(batch[index][])
    elif "model_description_embedding" in sample_keys: # sft data
        pass
    else:
        ValueError("No embeeding keys")
