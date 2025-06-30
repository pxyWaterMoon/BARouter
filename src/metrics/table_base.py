class TabelBasedModel:
    def __init__(self, dataset, budget):
        self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
        self.budget = budget
    
    def feedback(self, x, action):
        response = self.gt[x][action]["response"]
        cost = self.gt[x][action]["total_cost"]
        if self.budget < cost:
            return None, 0, 0
        else:
            reward = self.gt[x][action]["reward"]
            self.budget -= cost
            return response, reward, cost
        
        
        
        # return , self.gt[x][action]["reward"], self.gt[x][action]["total_cost"]