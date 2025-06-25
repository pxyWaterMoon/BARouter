class TabelBasedModel:
    def __init__(self, dataset):
        self.gt = {data["prompt"]: data["ground_truth"] for data in dataset}
    
    def feedback(self, x, action):
        return self.gt[x][action]["response"], self.gt[x][action]["reward"], self.gt[x][action]["total_cost"]