from src.datasets.simulerdata import SimulerDataset
from src.datasets.prompt_only import PromptOnlyDataset, PromptOnlyDataLoader
from src.datasets.sftdata import SFTDataset
import numpy as np
import pandas as pd
import os
from src.algorithms.routting_algorithms.AUPD import AUPD
from src.logger import Logger
from tqdm import tqdm
from src.algorithms.utils import embedding_sample

def run_system(T, env, agent, logger):
    print(f"Starting system run for {T} rounds...")
    with tqdm(total=T) as pbar_total:
        for t in range(T):
            sample = env.get_sample()
            action = agent.take_action(sample)
            if action is None:
                response = None
                reward = 0
                cost = 0
            else:
                response, reward, cost = env.feedback(sample, action)
            agent.update(reward, cost)
            logger.log_signal(sample["prompt"], action, reward, cost, t)
            logger.log_scalar(
                {
                    "train/current_reward": reward,
                    "train/current_cost": cost,
                    "train/average_reward": logger.get_log_value("rewards", range(t+1)),
                    "train/average_cost": logger.get_log_value("costs", range(t+1)),
                    "train/budget": env.current_budget,
                },
                step=t,
            )
            pbar_total.update(1)
    print(f"System run completed with {T} rounds.")
    return

def build_predictor_models(model_config, key):
    if model_config["type"] == "xgb":
        from src.algorithms.predictor.xgb import XGB
        model = XGB()
        SFT_dataset = SFTDataset(file_path=model_config["sft_file_path"])
        model.offline_training(SFT_dataset, key=key)
    elif model_config["type"] == "god":
        from src.algorithms.predictor.god import God
        model = God(
            file_path=model_config["file_path"],
            key=key,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    return model

def build_environment(env_config, budget, T):
    if env_config["type"] == "table":
        from src.envs.table_base import TabelBasedEnv
        simuler_dataset = SimulerDataset(file_path=env_config["file_path"])
        env_model = TabelBasedEnv(simuler_dataset, budget=budget)
        if env_model.support_length() < T:
            raise ValueError(f"The number of round table environment can support is {env_model.support_length()}, which is less than T={T}.")
    else:
        raise ValueError(f"Unsupported environment type: {env_config['type']}")
    return env_model

def build_agent(agent_config, B, T, logger):
    # build the predictor models
    if "rmodel" in agent_config.keys():
        rmodel = build_predictor_models(agent_config["rmodel"], key="reward")
    if "cmodel" in agent_config.keys():
        cmodel = build_predictor_models(agent_config["cmodel"], key="cost") # todo: change to cost (this need to change the data processing code)
    if agent_config["type"] == "AUPD":
        from src.algorithms.routting_algorithms.AUPD import AUPD
        agent = AUPD(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B, # todo: align it with the budget in AUPD code
            embedding_fn=embedding_sample,  # Function to embed the sample
            buffer_size=agent_config.get("buffer_size", 1024)
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_config['type']}")
    return agent


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
    agent = build_agent(config["agent"], B, T, logger)

    print(f"The system is constructed successfully!\n")

    # Run the system
    run_system(config["T"], env, agent, logger)
    
    print(f"System run completed. Logs saved to {logger_path}")
    logger.save_history()
    
    return


if __name__ == "__main__":
    from src.configs.read_config import argument_parser, load_config
    args = argument_parser()
    config = load_config(args)
    main(config)