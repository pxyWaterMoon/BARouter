
from src.data_processed import utils
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import os
from datasets import load_dataset

data_path = "./data/rawdata/sprout"
save_path = "./data/processed/bert-base-uncased/sprout_small_no4omini/"
model_path = "./models/bert-base-uncased"
cost_scale = 1000

# Load raw data
def process_sprout_data(split):
    train_rawdata = load_dataset("./data/rawdata/sprout", split=split)


    available_models_description = {
        'aws-claude-3-5-sonnet-v1': "A high-performance model designed for complex tasks, offering advanced reasoning and detailed responses suitable for professional use.", 
        'aws-titan-text-premier-v1': "A versatile and cost-effective model ideal for a wide range of applications, balancing performance and efficiency for everyday tasks.",
        'openai-gpt-4o': "A cutting-edge model delivering superior reasoning and creativity, perfect for intricate problem-solving and generating high-quality content.", 
        # 'openai-gpt-4o-mini': "A compact version of GPT-4o, providing efficient performance for less complex tasks while maintaining good quality in responses.",
        'wxai-granite-3-2b-instruct-8k-max-tokens': "A lightweight model optimized for quick responses and cost efficiency, suitable for straightforward queries and short-form content generation.",
        'wxai-granite-3-8b-instruct-8k-max-tokens': "A mid-sized model offering a balance between performance and cost, ideal for moderate complexity tasks and generating coherent responses.",
        'wxai-llama-3-1-70b-instruct': "A powerful large-scale model excelling in deep understanding and long-form content creation, suitable for advanced applications requiring high accuracy.",
        'wxai-llama-3-1-8b-instruct': "A robust model providing strong performance for a variety of tasks, balancing depth of understanding with efficiency for general-purpose use.",
        'wxai-llama-3-2-1b-instruct': "A compact model designed for fast and efficient responses, ideal for simple tasks and applications where speed is crucial.",
        'wxai-llama-3-2-3b-instruct': "A small yet capable model offering quick turnaround times and cost savings, suitable for basic queries and short interactions.",
        'wxai-llama-3-3-70b-instruct': "A top-tier model delivering exceptional performance in complex reasoning and detailed content generation, perfect for high-stakes applications.",
        'wxai-mixtral-8x7b-instruct-v01': "A highly efficient model tailored for multilingual tasks and technical discussions, providing balanced output length and quality.",
        'wxai-llama-3-405b-instruct': "A state-of-the-art model with unparalleled capabilities in understanding and generating human-like text, ideal for the most demanding applications."
    }

    cost_table = {
                'aws-claude-3-5-sonnet-v1': [3, 15], 'aws-titan-text-premier-v1': [0.5, 1.5],
                'openai-gpt-4o': [2.5, 10], 'openai-gpt-4o-mini': [0.15, 0.6],
                'wxai-granite-3-2b-instruct-8k-max-tokens': [0.1, 0.1],
                'wxai-granite-3-8b-instruct-8k-max-tokens': [0.2, 0.2],
                'wxai-llama-3-1-70b-instruct': [0.9, 0.9], 'wxai-llama-3-1-8b-instruct': [0.2, 0.2],
                'wxai-llama-3-2-1b-instruct': [0.06, 0.06], 'wxai-llama-3-2-3b-instruct': [0.06, 0.06],
                'wxai-llama-3-3-70b-instruct': [0.9, 0.9], 'wxai-mixtral-8x7b-instruct-v01': [0.6, 0.6],
                'wxai-llama-3-405b-instruct': [3.5, 3.5]
            }
    # Convert texts to embeddings
    deivce = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {deivce}")

    prompts_embeddings = utils.text2embeddings(
        train_rawdata["prompt"],
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
    for idx in tqdm(range(len(train_rawdata)), desc="Processing data"):
        rawdata = train_rawdata[idx]
        prompt = rawdata["prompt"]
        gt = {}
        for model in available_models_description.keys():
            model_gt = rawdata[model]
            gt_per_model = {
                "response": model_gt["response"],
                "reward": model_gt["score"],
                "cost": (model_gt["num_input_tokens"] * cost_table[model][0] / 1_000_000 + model_gt["num_output_tokens"] * cost_table[model][1] / 1_000_000) * cost_scale, # (TODO) A Bug?
            }
            gt[model] = gt_per_model
        processed_data = {
            "available_models_description": available_models_description,
            "available_models_description_embeddings": available_models_description_embeddings,
            "prompt": prompt,
            "prompt_embedding": prompts_embeddings[idx].numpy(),
            "gt": gt,
        }
        processed_dataset.append(processed_data)
    return processed_dataset

processed_dataset = []
for split in ["validation", "test"]:
    processed_dataset += process_sprout_data(split)

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


