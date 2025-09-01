
from src.data_processed import utils
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os

data_path = "./data/rawdata/sprout/sprout_raw.pkl"
save_path = "./data/processed/all-mpnet-base-v2/sprout/"
model_path = "./models/all-mpnet-base-v2"
cost_scale = 1/80.0

# Load raw data
rawdata = pd.read_pickle(data_path)
# print(rawdata.keys())
prompts = rawdata["prompt"].tolist()
print(f"Number of prompts: {len(prompts)}")
available_models_description = {
    'aws-claude-3-5-sonnet-v1':'aws-claude-3-5-sonnet-v1',
    'aws-titan-text-premier-v1':'aws-titan-text-premier-v1',
    'openai-gpt-4o':'openai-gpt-4o',
    'openai-gpt-4o-mini':'openai-gpt-4o-mini',
    'wxai-granite-3-2b-instruct-8k-max-tokens':'wxai-granite-3-2b-instruct-8k-max-tokens',
    'wxai-granite-3-8b-instruct-8k-max-tokens':'wxai-granite-3-8b-instruct-8k-max-tokens',
    'wxai-llama-3-1-70b-instruct':'wxai-llama-3-1-70b-instruct',
    'wxai-llama-3-1-8b-instruct':'wxai-llama-3-1-8b-instruct',
    'wxai-llama-3-2-1b-instruct':'wxai-llama-3-2-1b-instruct',
    'wxai-llama-3-2-3b-instruct':'wxai-llama-3-2-3b-instruct',
    'wxai-llama-3-3-70b-instruct':'wxai-llama-3-3-70b-instruct',
    'wxai-llama-3-405b-instruct':'wxai-llama-3-405b-instruct',
    'wxai-mixtral-8x7b-instruct-v01':'wxai-mixtral-8x7b-instruct-v01',
}

# Convert texts to embeddings
deivce = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {deivce}")

prompts_embeddings = utils.text2embeddings(
    prompts,
    model_path=model_path,
    batch_size=32,
    device=deivce
)
print(prompts_embeddings.shape)
modeldesc_embeddings = utils.text2embeddings(
    list(available_models_description.values()),
    model_path=model_path,
    batch_size=1,
    device=deivce
)
available_models_description_embeddings = {
    model: emb.numpy() for model, emb in zip(available_models_description.keys(), modeldesc_embeddings)
}

# get avialable models and write the model description
# print(rawdata_gb_sampleid.get_group("arc-challenge.val.136"))
processed_dataset = []
for index in tqdm(range(len(rawdata)), desc="Processing data"):
    df = rawdata.iloc[index]
    sample_id = df["sample_id"]
    sample_id = "routerbench_0shot." + str(sample_id)
    prompt = df["prompt"]
    gt = {}
    for model in available_models_description.keys():
        gt_per_model = {
            "response": df[model + "|model_response"],
            "reward": df[model],
            "cost": df[model + "|total_cost"] * cost_scale,
        }
        gt[model] = gt_per_model
    processed_data = {
        "sample_id": sample_id,
        "available_models_description": available_models_description,
        "available_models_description_embeddings": available_models_description_embeddings,
        "prompt": prompt,
        "prompt_embedding": prompts_embeddings[prompts.index(prompt)].numpy(),
        "gt": gt,
    }
    processed_dataset.append(processed_data)

# randomly split processed_data into train, test
np.random.seed(42)
np.random.shuffle(processed_dataset)
train_size = int(0.8 * len(processed_dataset))
train_dataset = processed_dataset[:train_size]
test_dataset = processed_dataset[train_size:]

# Save processed data as .parquet files
if not os.path.exists(save_path):
    os.makedirs(save_path)


train_df = pd.DataFrame(train_dataset)
train_df.to_parquet(os.path.join(save_path, "online.parquet" if cost_scale == 1.0 else f"online_cost_scale_{cost_scale}.parquet"), engine='pyarrow', index=False)
test_df = pd.DataFrame(test_dataset)
test_df.to_parquet(os.path.join(save_path, "offline.parquet" if cost_scale == 1.0 else f"offline_cost_scale_{cost_scale}.parquet"), engine='pyarrow', index=False)
print(f"Processed data saved to {save_path}")


