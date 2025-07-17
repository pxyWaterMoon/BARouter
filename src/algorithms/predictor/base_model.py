class BasePredictor():
    concatenate = True
    def __init__(self):
        pass
    
    def offline_training(self, dataset, key:str):
        raise NotImplementedError

    def online_update(self, X, y, global_step):
        raise NotImplementedError

    def predict(self, prompt):
        raise NotImplementedError