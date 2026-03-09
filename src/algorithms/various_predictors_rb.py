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
    # print(config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def read_dir(dir_path):
    name_list = os.listdir(dir_path)
    cfg_list = []
    for file_name in name_list:
        cfg = read_config(f"{dir_path}/{file_name}")
        cfg_list.append(cfg)
    return cfg_list
# config_list

def combine(cfg_env, cfg_agt, cfg_mdl, budget:int, proj_name:str):
    cfg_total = {}
    cfg_total.update(cfg_env)
    cfg_total.update(cfg_agt)
    cfg_total["agent"].update(cfg_mdl)
    cfg_total["budget"]=budget
    cfg_total["project_name"]=proj_name
    return cfg_total

if __name__ == "__main__":
    cfg_env_list = read_dir("./src/configs/predictors_rb_ratio/cfg_env")
    cfg_agt_list = read_dir("./src/configs/predictors_rb_ratio/cfg_agent")
    cfg_mdl_list = read_dir("./src/configs/predictors_rb_ratio/cfg_model")
    # cfg_total = combine(cfg_env_list[0], cfg_agt_list[0], cfg_mdl_list[0],1600,"vp_test")
    # print(cfg_total)
    # print(main(cfg_total))

    B_list = [2000,5000,10000]

    carrot_mu={
        2000:0.98,
        5000:0.8,
        10000:0.25
    }

    # test_agent_list = ["carrot2", "AUPD_exp"]
    # test_model_list = [""]

    repeat = 5
    res = {}
    for cfg_env in cfg_env_list:
        for cfg_mdl in cfg_mdl_list:
            for cfg_agt in cfg_agt_list:
                for budget in B_list:
                    name = f'{cfg_agt["agent"]["type"]}_{cfg_mdl["rmodel"]["type"]}_B={budget}'

                    if "carrot" in cfg_agt["agent"]["type"]:
                        cfg_agt["agent"]["mu"] = carrot_mu[budget]

                    cfg_total = combine(cfg_env,cfg_agt,cfg_mdl,budget,name)
                    res[name]=[]
                    for _ in range(repeat):
                        avg_r = main(cfg_total)
                        res[name].append(avg_r)
                        print(avg_r)
            # break

    import json
    with open(f"./outputs/predictors/1006_rb.json", "w") as f:
        json.dump(res, f)
                    
    
