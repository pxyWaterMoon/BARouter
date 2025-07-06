import torch
import numpy as np
from src.datasets.prompt_only import PromptOnlySample

def embedding_sample(sample: PromptOnlySample, key: str | None = None, concatenate: bool = True):
    # print(batch)
    
    available_models = list(sample["available_models_description"].keys())
    k = len(available_models)
    x = np.array([sample["prompt_embedding"] for _ in range(k)])  # (K, dx)
    a = [sample["available_models_description_embeddings"][model_name] for model_name in available_models]
    if key is not None:
        gt = sample["ground_truth"]
        if key not in gt.keys():
            raise ValueError(f"{key} not in ground truth")
        y = [gt[model_name][key] for model_name in available_models]
    if concatenate:
        if key is not None:
            return np.concatenate([x, a], axis=-1), np.array(y)
        else:
            return np.concatenate([x, a], axis=-1)
    else:
        if key is not None:
            return np.array(x), np.array(a), np.array(y)
        else:
            return np.array(x), np.array(a)
        

def embedding_batch(batch: list, key: str | None = None, concatenate: bool = True):
    bs = len(batch)
    # print(batch)
    sample_keys = batch[0].keys()
    if concatenate:
        x = []
    else:
        x = []
        a = []
    if key is not None:
        y = []
    if "available_models_description_embeddings" in sample_keys: # need to extend prompt size with available models description size
        for index in range(bs):
            sample = batch[index]
            sample_x = []
            if not concatenate:
                sample_a = []
            if key is not None:
                sample_y = []
            available_models = list(sample["available_models_description"].keys())
            for model_name in available_models:
                if concatenate:
                    sample_x.append(np.concatenate([sample["prompt_embedding"], sample["available_models_description_embeddings"][model_name]]))
                else:
                    sample_x.append(sample["prompt_embedding"])
                    sample_a.append(sample["available_models_description_embeddings"][model_name])
                if key is not None:
                    gt = sample["ground_truth"][model_name]
                    if key not in gt.keys():
                        raise ValueError(f"{key} not in ground truth")
                    sample_y.append(gt[key])
            x.append(sample_x)
            if not concatenate:
                a.append(sample_a)
            if key is not None:
                y.append(sample_y)
            if concatenate:
                if key is not None:
                    return np.array(x), np.array(y) # x.shape = (bs, K, dx + da), y.shape = (bs, K)
                else:
                    return np.array(x) # x.shape = (bs, K, dx + da)
            else:
                if key is not None:
                    return np.array(x), np.array(a), np.array(y) # x.shape = (bs, K, dx), a.shape = (bs, K, da), y.shape = (bs, K)
                else:
                    return np.array(x), np.array(a) # x.shape = (bs, K, dx), a.shape = (bs, K, da)
    elif "model_description_embedding" in sample_keys: # no need to extend
        # batch is a list of dictionaries with keys: prompt_embedding, model_description_embedding, reward, cost
        x = np.array([data["prompt_embedding"] for data in batch])
        a = np.array([data["model_description_embedding"] for data in batch])
        if key is not None:
            if not all (key in data.keys() for data in batch):
                raise ValueError(f"{key} not in all samples")
            y = np.array([data[key] for data in batch])
        if concatenate:
            return np.concatenate([x, a], axis=-1) if key is None else (np.concatenate([x, a], axis=-1), y)
        else:
            return (x, a) if key is None else (x, a, y)
    else:
        ValueError("No embedding keys")


# test embedding batch function
from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset, PromptOnlyDataLoader
from src.datasets.sftdata import SFTDataset, SFTDataLoader, SFTBufferPool
if __name__ == "__main__":
    simuler_dataset = SimulerDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/train.parquet")
    SFT_dataset = SFTDataset(file_path="data/processed/all-mpnet-base-v2/routerbench_0shot/test.parquet")

    sft_loader = SFTDataLoader(SFT_dataset, batch_size=4, shuffle=True)
    
    for sft_batch in sft_loader:
        x, a = embedding_batch(sft_batch, concatenate=False)
        print(x.shape, a.shape)
        print(embedding_batch(sft_batch, concatenate=True).shape)
        x, a, y = embedding_batch(sft_batch, "reward", False)
        print(x.shape, a.shape, y.shape)
        x, y = embedding_batch(sft_batch, "reward", True)
        print(x.shape, y.shape)
        break

    promptonly_dataset = PromptOnlyDataset(simuler_dataset)
    promptonly_loader = PromptOnlyDataLoader(promptonly_dataset)
    for promptonly_batch in promptonly_loader:
        print(embedding_batch(promptonly_batch, concatenate=True).shape)
        x, a = embedding_batch(promptonly_batch, concatenate=False)
        print(x.shape, a.shape)
        break

    buffferpool = SFTBufferPool()

    for sft_batch in sft_loader:
        buffferpool.add(sft_batch[0])
    
    update_batch = buffferpool.get_batch()
    print(embedding_batch(update_batch).shape)
    

    

