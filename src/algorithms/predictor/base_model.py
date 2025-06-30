class BasePredictor():
    def __init__(self):
        pass
    
    def offline_training(self):
        raise NotImplementedError

    def online_update(self, X, y):
        raise NotImplementedError

    def predict(self, prompt):
        raise NotImplementedError