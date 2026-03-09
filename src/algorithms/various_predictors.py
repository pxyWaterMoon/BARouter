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
    total_reward = 0
    print(f"Starting system run for {T} rounds...")
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
            total_reward += reward
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
    return total_reward / T


def main(config):
    from src.algorithms.main import build_environment, build_agent
    print(f"Building system with configuration: {config}")
    
    B = config["budget"]
    T = config["T"]
    seed = config.get("seed", 42)
    # Initialize logger
    logger_filename = f"{config['project_name']}"
    logger_path = os.path.join(config["log_dir"], logger_filename)
    logger = Logger(logger_path)
    
    # Build the environment
    env = build_environment(config["environment"], B, T, seed)

    # Build the agent
    agent = build_agent(config["agent"], B, T, logger, env.action_space)

    print(f"The system is constructed successfully!\n")

    # Run the system
    res = run_system(config["T"], env, agent, logger)
    
    print(f"System run completed. Logs saved to {logger_path}")
    logger.save_history()
    
    return res

def read_config(config_path):
    import yaml
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    
    # args = argument_parser()
    # config = load_config(args)
    # path = "src/configs/exp1/rb/B=2000/AUPD_knn.yaml"
    # main(read_config(path))

    # group = {
    #     # "rb2":[2000,5000,20000],
    #     "sp":[1600]
    # }
    respeat = 5
    res = {}
    b_list = [1500]
    for budget in b_list:
        folder = f"src/configs/predictors/B={budget}"
        cfg_list = os.listdir(folder)
        for cfg_path in cfg_list:
            if "knn_on" not in cfg_path:
                continue
            cfg = read_config(f"{folder}/{cfg_path}")
            cfg["budget"] = budget
            res_list = []
            for _ in range(respeat):
                res_list.append(main(cfg))
            print(np.average(np.array(res_list)))
            name = cfg_path
            if name not in res:
                res[name]=[]
            res[name].append(res_list)
    
    import json
    with open(f"./outputs/predictors/B=1500_on.json", "w") as f:
        json.dump(res, f)