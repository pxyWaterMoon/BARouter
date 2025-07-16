import json
import sys
from tqdm import tqdm
from vllm import SamplingParams, LLM
import pandas as pd
import argparse
import os

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
    
def generate_prompt(question):
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    return question + " " + instruction_following

def main(args):

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

    dataset = load_dataset(args.data_path)
    print(f"Loaded {len(dataset)} problems from {args.data_path}")

    # 生成提示词
    prompts = [generate_prompt(item["question"]) for item in dataset]

    # 批量推理
    results = []
    batch_size = args.batch_size
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        
        for j, output in enumerate(outputs):
            # total_tokens = output.prompt_tokens + output.outputs[0].tokens
            results.append({
                # "id": dataset[i+j]["id"],
                "question": dataset[i+j]["question"],
                "ground_truth": dataset[i+j]["answer"],
                "completion": output.outputs[0].text.strip(),
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            })
    # 保存结果
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.save_path}")

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
        default=2048,
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