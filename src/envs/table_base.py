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
from src.datasets.simulerdata import SimulerDataLoader
from src.datasets.prompt_only import PromptOnlySample
class TabelBasedEnv(BaseEnv):
    def __init__(self, dataset, budget):
        super().__init__()
        self.loader = SimulerDataLoader(dataset,shuffle=True)
        self.total_budget = budget
        self.action_space = self.loader.get_action_space()
        self.reset()

    def reset(self):
        self.current_sample = None
        self.current_budget = self.total_budget
        self.loader.reset()
        
    def get_sample(self):
        self.current_sample = self.loader.get_sample()
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
        return len(self.loader)
    
        
    