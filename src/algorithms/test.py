from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset, PromptOnlyDataLoader
from src.datasets.sftdata import SFTDataset
import numpy as np
import pandas as pd
import os
from src.algorithms.routting_algorithms.AUPD import AUPD
from src.algorithms.routting_algorithms.AUPD_exp import AUPD_exp
from src.logger import Logger
from tqdm import tqdm


def run_system(T, env, agent, logger):
    print(f"Starting system run for {T} rounds...")
    total_r = 0
    with tqdm(total=T) as pbar_total:
        for t in range(T):
            sample = env.get_sample()
            action = agent.take_action(sample)
            if action == "None":
                response = None
                reward = 0
                cost = 0
            else:
                response, reward, cost = env.feedback(sample, action)

            total_r += reward
            log_sample = agent.update(reward, cost, response)
            logger.log_signal(log_sample, t)
            logger.log_scalar(
                {
                    "train/current_reward": reward,
                    "train/current_cost": cost,
                    "train/average_reward": logger.get_log_value("reward", range(t+1)),
                    "train/average_cost": logger.get_log_value("cost", range(t+1)),
                    "train/budget": env.current_budget,
                },
                step=t,
            )
            pbar_total.update(1)
    print(f"System run completed with {T} rounds.")
    return total_r/T


def main(config):
    print(f"Building system with configuration: {config}")
    
    B = config["budget"]
    T = config["T"]
    # Initialize logger
    logger_filename = f"{config['project_name']}"
    logger_path = os.path.join(config["log_dir"], logger_filename)
    logger = Logger(logger_path)
    
    # Build the environment
    env = build_environment(config["environment"], B, T)

    # Build the agent
    agent = build_agent(config["agent"], B, T, logger, env.action_space)

    print(f"The system is constructed successfully!\n")

    # Run the system
    r = run_system(config["T"], env, agent, logger)
    
    print(f"System run completed. Logs saved to {logger_path}")
    logger.save_history()
    
    return r

def read_config(config_path):
    import yaml
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    from src.algorithms.main import build_agent,build_environment
    env_path = "/home/zulk2024/ICLR2026/tmp_configs/test_env.yaml"
    agent_path_list = [
        "/home/zulk2024/ICLR2026/src/configs/test1/AUPD_routerbench_knn.yaml",
        "/home/zulk2024/ICLR2026/src/configs/test1/cons_routerbench.yaml",
    ]
    logger_path = "outputs/log0919"
    B_list = [500,1000,1500]
    env_cfg = read_config(env_path)
    T = env_cfg["T"]
    env_cfg = env_cfg["environment"]

    res = {}

    for agent_path in agent_path_list:
        agent_cfg = read_config(agent_path)
        title = agent_cfg['agent']['type']
        if title in res.keys():
            title = title + "_1"
        res[title]=[]
        for B in B_list:
            env = build_environment(env_cfg, B, T)
            project_name = f"{agent_cfg['agent']['type']}_B={B}"
            logger = Logger(os.path.join(logger_path, project_name))
            agent = build_agent(agent_cfg["agent"], B, T, logger, env.action_space)
            r = run_system(T, env, agent, logger)
            print(project_name,r)
            res[title].append(r)
    
    import json
    name = "test"
    with open(f"./outputs/test_json/{name}_batch_results.json", "w") as f:
        json.dump(res, f)
