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
    return

def build_predictor_models(model_config, key, action_space, logger):
    if model_config["type"] == "xgbregressor":
        from src.algorithms.predictor.xgbregressor import XGBRegressorPredictor
        SFT_dataset = SFTDataset(file_path=model_config["sft_file_path"])
        model = XGBRegressorPredictor(SFT_dataset=SFT_dataset, key=key, offline= model_config["offline"])
    elif model_config["type"] == "xgbclassifier":
        from src.algorithms.predictor.xgbclassifier import XGBClassifierPredictor
        SFT_dataset = SFTDataset(file_path=model_config["sft_file_path"])
        model = XGBClassifierPredictor(SFT_dataset=SFT_dataset, key=key, offline= model_config["offline"])
    elif model_config["type"] == "mean":
        from src.algorithms.predictor.mean import Mean
        SFT_dataset = SFTDataset(file_path=model_config["sft_file_path"])
        model = Mean(SFT_dataset=SFT_dataset, key=key, offline= model_config["offline"])
    elif model_config["type"] == "god":
        from src.algorithms.predictor.god import God
        simuler_dataset = SimulerDataset(file_path=model_config["file_path"])
        model = God(
            dataset=simuler_dataset,
            key=key,
        )
    elif model_config["type"] == "mf":
        from src.algorithms.predictor.mf import MatrixFactorizationPredictor
        
        SFT_dataset = None if model_config["sft_file_path"] == "None" else SFTDataset(file_path=model_config["sft_file_path"])
        # if not model_config["offline"]:
        #     print("Online!!")
        #     SFT_dataset = None
        model = MatrixFactorizationPredictor(
            model_list=action_space,
            key=key,
            dim=model_config.get("dim", 128),
            text_dim=model_config.get("text_dim", 768),  #75,768
            offline_lr=model_config.get("offline_lr", 0.001),
            offline_epoch=model_config.get("offline_epoch", 10),
            online_lr=model_config.get("online_lr", 0.01),
            buffer_size=model_config.get("buffer_size", 64),
            online_decay=model_config.get("online_decay", 0.99),
            SFT_dataset=SFT_dataset,
            logger=logger,
        )
    elif model_config["type"] == "kmeans":
        from src.algorithms.predictor.kmeans import K_means
        simuler_dataset = SimulerDataset(file_path=model_config["file_path"])
        model = K_means(simuler_dataset,key=key,k=model_config.get("k",5))
    elif model_config["type"] == "kmeans_ucb":
        from src.algorithms.predictor.kmeans_ucb import K_means
        simuler_dataset = SimulerDataset(file_path=model_config["file_path"])
        model = K_means(simuler_dataset,key=key,k=model_config.get("k",5),c=0)
    elif model_config["type"] == "kmeans_beta":
        from src.algorithms.predictor.kmeans_beta import K_means
        simuler_dataset = SimulerDataset(file_path=model_config["file_path"])
        model = K_means(simuler_dataset,key=key,k=model_config.get("k",5))
    elif model_config["type"] == "kmeans_upd":
        from src.algorithms.predictor.online_kmeans import K_means_online
        model = K_means_online(key=key,n_action=len(action_space),n_cluters=model_config.get("k",20))
    elif model_config["type"] == "hf_model":
            from src.algorithms.predictor.hf_model import HFMoodelPredictor
            model = HFMoodelPredictor(
                model_list=action_space,
                key= model_config.get("key", key),
                model_name_or_path=model_config["model_name_or_path"],
                cost_table = model_config.get("cost_table", None),
                input_counter_path_or_name = model_config.get("input_counter_path_or_name", None),
            )
    elif model_config["type"] == "knn":
        from src.algorithms.predictor.knn import KNN
        simuler_dataset = SimulerDataset(file_path=model_config["file_path"])
        model = KNN(simuler_dataset, key=key, k=model_config.get("k", 100))
    elif model_config["type"] == "olknn":
        from src.algorithms.predictor.olknn import OLKNN
        simuler_dataset = SimulerDataset(file_path=model_config["file_path"])
        model = OLKNN(simuler_dataset, key=key, k=model_config.get("k", 100),offline=model_config.get("offline", True))
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    return model

def build_environment(env_config, budget, T, seed=42):
    if env_config["type"] == "table":
        from src.envs.table_base import TabelBasedEnv
        simuler_dataset = SimulerDataset(file_path=env_config["file_path"])
        env_model = TabelBasedEnv(simuler_dataset, budget=budget)
    elif env_config["type"] == "table_multistage_random":
        from src.envs.table_random import TabelMultistageRandomEnv
        simuler_datasets = [SimulerDataset(file_path=path) for path in env_config["file_paths"]]
        env_model = TabelMultistageRandomEnv(simuler_datasets, budget=budget, stages=env_config["stages"], T=T, seed=seed)
    elif env_config["type"] == "table_timevarious_random":
        from src.envs.table_random import TabelTimevariousRandomEnv
        simuler_datasets = [SimulerDataset(file_path=path) for path in env_config["file_paths"]]
        env_model = TabelTimevariousRandomEnv(simuler_datasets, budget=budget, stages=env_config["stages"], T=T, seed=seed)
    elif env_config["type"] == "server":
        from src.envs.server_base import ServerBasedEnv
        dataset = PromptOnlyDataset(file_path=env_config["data_path"])
        embedder_config = env_config.get("embedder", None)
        embedder=None
        if embedder_config is not None:
            if embedder_config["type"] == "SentenceTransformerEmbedder":
                 from src.embed import SentenceTransformerEmbedder
                 embedder = SentenceTransformerEmbedder(embedder_config["model_path"])
            else:
                raise ValueError(f"Unsupported text embedding type: {embedder_config['type']}")
        if env_config["reward_fn"] == "str_cmp":
            from src.online_judgement import str_cmp
        else:
            raise ValueError(f"Unsupported reward function type: {env_config['reward_fn']}")
        env_model = ServerBasedEnv(dataset, budget, env_config["model_info"], embedder.embed, str_cmp)
    else:
        raise ValueError(f"Unsupported environment type: {env_config['type']}")
    if env_model.support_length() < T:
        raise ValueError(f"The number of round table environment can support is {env_model.support_length()}, which is less than T={T}.")
    return env_model

def select_embedding_fn(name):
    if name == "sample2given_embedding":
        from src.algorithms.utils import sample2given_embedding
        return sample2given_embedding
    elif name == "sample2prompt":
        from src.algorithms.utils import sample2prompt
        return sample2prompt
    else:
        raise ValueError(f"Unsupported embbeding function: {name}")

def build_agent(agent_config, B, T, logger, action_space):
    # build the predictor models
    if "rmodel" in agent_config.keys():
        rmodel = build_predictor_models(agent_config["rmodel"], key="reward", action_space=action_space, logger=logger)
    if "cmodel" in agent_config.keys():
        cmodel = build_predictor_models(agent_config["cmodel"], key="cost", action_space=action_space, logger=logger)
    if agent_config["type"] == "AUPD":
        from src.algorithms.routting_algorithms.AUPD import AUPD
        agent = AUPD(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
            buffer_size=agent_config.get("buffer_size", 1024),
            v_scale=agent_config["v_scale"],
            allow_null=agent_config["allow_null"]
        )
    elif agent_config["type"] == "AUPD_exp":
        from src.algorithms.routting_algorithms.AUPD_exp import AUPD_exp
        agent = AUPD_exp(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
            buffer_size=agent_config.get("buffer_size", 1024),
            v_scale=agent_config["v_scale"],
            allow_null=agent_config["allow_null"],
            eta=agent_config["eta"] if "eta" in agent_config else 30
        )
    elif agent_config["type"] == "LOE2D":
        from src.algorithms.routting_algorithms.LOE2D import LOE2D
        agent = LOE2D(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            K=len(action_space),
            U = agent_config.get("U", 30),
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
            buffer_size=agent_config.get("buffer_size", 64)
        )
    elif agent_config["type"] == "FixAction":
        from src.algorithms.routting_algorithms.fix_action import FixAction
        if "action" not in agent_config.keys():
            raise ValueError("FixAction agent requires an 'action' key in the configuration.")
        action = agent_config["action"]
        agent = FixAction(action=action, T=T, logger=logger)
    elif agent_config["type"] == "google":
        from src.algorithms.routting_algorithms.google import GG
        agent = GG(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
            lam=agent_config.get("lambda", 0.1)
        )
    elif agent_config["type"] == "carrot2":
        from src.algorithms.routting_algorithms.carrot2 import Carrot2
        agent = Carrot2(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
            mu=agent_config.get("mu", 0.3)
        )
    elif agent_config["type"] == "ratio":
        from src.algorithms.routting_algorithms.ratio import Ratio
        agent = Ratio(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
        )
    elif agent_config["type"] == "cons":
        from src.algorithms.routting_algorithms.cons import Cons
        agent = Cons(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
        )
    elif agent_config["type"] == "cons2":
        from src.algorithms.routting_algorithms.cons2 import Cons2
        agent = Cons2(
            rmodel=rmodel,
            cmodel=cmodel,
            logger=logger,
            T=T,
            budget=B,
            embedding_fn=select_embedding_fn(agent_config["embedding_fn"]),  # Function to embed the sample
        )
    elif agent_config["type"] == "carrot":
        from src.algorithms.routting_algorithms.carrot import CarrotRouter
        agent = CarrotRouter(budget=B,mu=agent_config["mu"])
    elif agent_config["type"] == "gradient":
        from src.algorithms.routting_algorithms.gradient_policy import Gradient
        from src.datasets.simulerdata import SimulerDataset
        dataset = SimulerDataset(file_path=agent_config["offline_data"])
        agent = Gradient(dataset=dataset,budget_perround=B/T,k=20)
    else:
        raise ValueError(f"Unsupported agent type: {agent_config['type']}")
    return agent


def main(config):
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
    run_system(config["T"], env, agent, logger)
    
    print(f"System run completed. Logs saved to {logger_path}")
    logger.save_history()
    
    return logger.history


if __name__ == "__main__":
    from src.configs.read_config import argument_parser, load_config
    args = argument_parser()
    config = load_config(args)
    main(config)