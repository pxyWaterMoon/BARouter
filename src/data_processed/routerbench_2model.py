
from src.data_processed import utils
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os

data_path = "./data/rawdata/routerbench/routerbench_0shot.pkl"
save_path = "./data/processed/all-mpnet-base-v2/routerbench_0shot_2model/"
model_path = "./models/all-mpnet-base-v2"
cost_scale = 1000

# Load raw data
rawdata = pd.read_pickle(data_path)
# print(rawdata.keys())
prompts = rawdata["prompt"].tolist()
print(f"Number of prompts: {len(prompts)}")
available_models_description = {
    # "gpt-3.5-turbo-1106": "A lightweight and cost-efficient model ideal for everyday tasks, generating concise responses with moderate reasoning capabilities.", 
    # "claude-instant-v1": "An affordable model providing quick, straightforward answers best suited for simple queries and short-text generation tasks.",
    # "claude-v1": "A moderately-priced model with enhanced reasoning skills, balancing depth and efficiency for extended conversations.",
    # "claude-v2": "A powerful premium model excelling in complex analysis and long-form content creation with strong contextual understanding.",
    "gpt-4-1106-preview": "A top-tier expensive model delivering exceptional reasoning and creative capabilities for advanced problem-solving.",
    # "meta/llama-2-70b-chat": "A robust open-source model offering strong general-purpose performance at no cost, great for diverse conversational needs.",
    # "mistralai/mixtral-8x7b-chat": "A highly efficient open-source model specialized in multilingual tasks and technical discussions with balanced output length.",
    # "zero-one-ai/Yi-34B-Chat": "A capable bilingual model freely handling both English and Chinese content with mid-length analytical responses.",
    "WizardLM/WizardLM-13B-V1.2": "A free specialized model optimized for complex instruction-following and detailed multi-turn dialogues.",
    # "meta/code-llama-instruct-34b-chat": "A purpose-built coding model freely providing detailed technical explanations and extended code solutions.",
    # "mistralai/mistral-7b-chat": "A compact open-source model delivering fast, focused responses perfect for lightweight applications."
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
    sample_id = "routerbench_0shot." + sample_id
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


