from src.envs.base_env import BaseEnv
from src.datasets.simulerdata import SimulerDataLoader
from src.datasets.prompt_only import PromptOnlySample
from openai import OpenAI

class ServerBasedEnv(BaseEnv):
    def __init__(self, dataset, budget, model_info):
        super().__init__()
        self.loader = SimulerDataLoader(dataset)
        self.total_budget = budget
        self.action_space = list(model_info.keys())
        self.model_clients = {}
        for model_name in self.action_space:
            self.model_clients[model_name] = OpenAI(
                api_key=model_info[model_name]["api_key"],
                base_url=model_info[model_name]["base_url"],
            )
        self.moldel_costs = {model_name: model_info[model_name]["cost_per_token"] for model_name in self.action_space}
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
        if self.current_budget <= 0:
            return None, 0, 0
        output = self.model_clients[action].chat.completions.create(
            prompt=sample["prompt"],
        )
        response = output.choices[0].message.content
        cost = output.usage.total_tokens * self.moldel_costs[action]
        reward = 1 # todo: Implement a proper reward function based on the response
        self.current_budget -= cost
        return response, reward, cost
    
    def support_length(self):
        """
        Return the maximum number of rounds the environment can support.
        
        If the environment does not have a limit, return None.
        """
        return len(self.loader)