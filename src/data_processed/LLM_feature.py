import json
import sys
from tqdm import tqdm
from vllm import SamplingParams, LLM
import pandas as pd
import argparse
import numpy as np

# 加载数据集
def load_dataset(file_path):
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path).to_dict(orient='records')
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path).to_dict(orient='records')
    else:
        raise ValueError("Unsupported file format.")
    
# def generate_prompt(question):
#     instruction_following = 'Let\'s think step by step and output the final answer after "####".'
#     return question + " " + instruction_following



def format_chat_prompt(message):
    return f"""<|system|>
You are a helpful AI assistant.</s>
<|user|>
{message}</s>
<|assistant|>
"""

def create_feature_extraction_prompt(question_prompt:str, property_question:str):
    prompt = f"""
Analyze the following question and determine if it meets the specified property.
The question will begin at 'BEGIN QUESTION' and end at 'END QUESTION'

BEGIN QUESTION

{question_prompt}

END QUESTION

PROPERTY TO CHECK: {property_question}

Your response should be "yes" or "no" in lower case only, no explanations.


"""
    return format_chat_prompt(prompt.strip())

def norm(value_list):
    max_value = np.max(value_list)
    return np.array(value_list).reshape(-1,1)/max_value

def json2emb(json_list):
    feature_type = \
    {
        "complexity_level": "norm", # 归一化
        "question_type": "one_hot",
        "question_format": "one_hot",
        "ambiguity_score": "norm",
        "reasoning_depth": "norm",
        "specificity": "norm",
        "estimated_difficulty": "norm",
        "clarity_score": "norm"
    }
    for key,method in feature_type.items():
        value_list = [data[key] for data in json_list]
        if method == "norm":
            feature = norm(value_list)
            print(feature.shape)
            exit(0)
    return []

def save_str(str_list, path):
    json.dump(str_list, open(path, "w"))

def main(args):
    data_path = args.data_path
    dataset = load_dataset(data_path)
    # print(dataset[0].keys())
    # exit(0)

    # 配置采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.response_length,
    )

    # 初始化分布式LLM
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.prompt_length + args.response_length,
        gpu_memory_utilization=0.8,
        swap_space=16  # 启用磁盘交换
    )
    # print(len(dataset))
    # exit(0)
    # N = 100
    # dataset = dataset[:N]

    questions = [data["prompt"] for data in dataset]
    property_question = "Does the prompt relate to real-world applications?"

    prompts = [create_feature_extraction_prompt(question,property_question) for question in questions]


    str_list = []

    for i in tqdm(range(len(prompts))):

        prompt = prompts[i]
        if len(prompt) > args.prompt_length:
            str_list.append("yes")
            continue
        outputs = llm.generate(prompt, sampling_params)
        output_str = outputs[0].outputs[0].text
        str_list.append(output_str)

    
    pro_list = []
    for str_item in str_list:
        pro_list.append(1 if "yes".casefold() in str_item.casefold() else 0)
    
    path = "/home/zulk2024/ICLR2026/data/processed/bert-base-uncased/sprout_test/pro7.json"
    save_str(pro_list,path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate answers for a dataset using a specified model.',
    )
    parser.add_argument(
        '--num_sequences',
        type=int,
        default=1,
        help='Number of answers generated per prompt.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling.',
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for generation.',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the dataset file.',
        required=True,
    )
    parser.add_argument(
        '--prompt_length',
        type=int,
        default=1024,
        help='Maximum length of the model prompt.',
    )
    parser.add_argument(
        '--response_length',
        type=int,
        default=1024,
        help='Maximum length of the model response.',
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='Path or name of the model to use for generation.',
        required=True,
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Size of tensor parallelism for distributed inference.',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='File to save the generated predictions.',
    )
    
    main(parser.parse_args())