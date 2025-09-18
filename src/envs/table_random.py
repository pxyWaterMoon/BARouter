# class TabelBasedEnv:
#     def __init__(self, dataset, budget):
#         self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
#         self.budget = budget
    
#     def feedback(self, x, action):
#         response = self.gt[x][action]["response"]
#         cost = self.gt[x][action]["cost"]
#         if self.budget < cost:
#             return None, 0, 0
#         else:
#             reward = self.gt[x][action]["reward"]
#             self.budget -= cost
#             return response, reward, cost
        
        
        
        # return , self.gt[x][action]["reward"], self.gt[x][action]["cost"]
    
from src.envs.base_env import BaseEnv
from src.datasets.simulerdata import RandomSimulerDataLoader
from src.datasets.prompt_only import PromptOnlySample
import random
class TabelMultistageRandomEnv(BaseEnv):
    def __init__(self, datasets, budget, stages, T, seed): # stages [(float (ratio), [float, ... (level ratio)])]
        super().__init__()
        self.loaders = [RandomSimulerDataLoader(dataset) for dataset in datasets]
        self.total_budget = budget
        self.action_space = self.loaders[0].get_action_space()
        # check the action space is the same
        if not all(loader.get_action_space() == self.action_space for loader in self.loaders):
            raise ValueError("Inconsistent action spaces detected.")
        self.stages = stages
        self.seed = seed
        self.T = T
        # check all the stage have the same number of weights as the number of loaders
        self.reset()

    def reset(self):
        random.seed(self.seed)
        self.current_sample = None
        self.current_budget = self.total_budget
        self.current_t = 0
        self.current_stage = 0
        self.current_T_ratio, self.current_level_ratio = self.stages[0] 
        for loader in self.loaders:
            loader.reset()
        
    def get_sample(self):
        level = random.choices(range(len(self.loaders)), weights=self.current_level_ratio)[0]
        self.current_sample = self.loaders[level].get_sample()
        self.current_t += 1
        if self.current_t >= self.T * self.current_T_ratio and self.current_t < self.T:
            self.current_stage += 1
            new_T_ratio, self.current_level_ratio = self.stages[self.current_stage]
            self.current_T_ratio += new_T_ratio
        return PromptOnlySample(
            prompt=self.current_sample["prompt"],
            prompt_embedding=self.current_sample["prompt_embedding"],
            available_models_description=self.current_sample["available_models_description"],
            available_models_description_embeddings=self.current_sample["available_models_description_embeddings"]
        )
        
    def feedback(self, sample, action):
        if sample["prompt"] != self.current_sample["prompt"]:
            raise ValueError("Sample prompt does not match current sample. please call get_sample() first.")
        gt = self.current_sample["ground_truth"]
        response = gt[action]["response"]
        cost = gt[action]["cost"]
        if self.current_budget < cost:
            return None, 0, 0
        else:
            reward = gt[action]["reward"]
            self.current_budget -= cost
            return response, reward, cost
    
    def support_length(self):
        """
        Return the maximum number of rounds the environment can support.
        
        If the environment does not have a limit, return None.
        """
        # return len(self.loader)
        return self.T
    
        
class TabelTimevariousRandomEnv(BaseEnv):
    def __init__(self, datasets, budget, stages, T, seed): # stages [(float (ratio), [float, ... (level ratio)])]
        super().__init__()
        self.loaders = [RandomSimulerDataLoader(dataset) for dataset in datasets]
        self.total_budget = budget
        self.action_space = self.loaders[0].get_action_space()
        # check the action space is the same
        if not all(loader.get_action_space() == self.action_space for loader in self.loaders):
            raise ValueError("Inconsistent action spaces detected.")
        self.stages = stages
        self.seed = seed
        self.T = T
        # check all the stage have the same number of weights as the number of loaders
        self.reset()

    def reset(self):
        random.seed(self.seed)
        self.current_sample = None
        self.current_budget = self.total_budget
        self.current_t = 0
        self.current_stage = 0
        p = self.stages[0] 
        self.current_level_ratio = [p, 1 - p]
        for loader in self.loaders:
            loader.reset()
        
    def get_sample(self):
        level = random.choices(range(len(self.loaders)), weights=self.current_level_ratio)[0]
        self.current_sample = self.loaders[level].get_sample()
        self.current_t += 1
        delta = self.current_t / self.T * (self.stages[0] - self.stages[1])
        p = self.stages[0] - delta
        self.current_level_ratio = [p, 1 - p]
        return PromptOnlySample(
            prompt=self.current_sample["prompt"],
            prompt_embedding=self.current_sample["prompt_embedding"],
            available_models_description=self.current_sample["available_models_description"],
            available_models_description_embeddings=self.current_sample["available_models_description_embeddings"]
        )
        
    def feedback(self, sample, action):
        if sample["prompt"] != self.current_sample["prompt"]:
            raise ValueError("Sample prompt does not match current sample. please call get_sample() first.")
        gt = self.current_sample["ground_truth"]
        response = gt[action]["response"]
        cost = gt[action]["cost"]
        if self.current_budget < cost:
            return None, 0, 0
        else:
            reward = gt[action]["reward"]
            self.current_budget -= cost
            return response, reward, cost
    
    def support_length(self):
        """
        Return the maximum number of rounds the environment can support.
        
        If the environment does not have a limit, return None.
        """
        # return len(self.loader)
        return self.T