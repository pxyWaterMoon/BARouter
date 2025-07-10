import pandas as pd
import numpy as np
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch
import torch.nn as nn
import torch
from src.datasets.sftdata import SFTDataset, SFTDataLoader

class MatrixFactorization(nn.Module):
    def __init__(self,
                 num_models,
                 dim,
                 text_dim,
                 ):
        self.model_embed = nn.Embedding(num_models, dim)
        self.text_projection = nn.Linear(text_dim, dim, bias=False)
        self.classifier = nn.Linear(dim, 1, bias=False)  # Assuming binary classification for reward/cost prediction
        
    def forward(self, model_id, prompt_embedding):
        model_embedding = self.model_embed(model_id)
        text_embedding = self.text_projection(prompt_embedding)
        prediction = self.classifier(model_embedding * text_embedding)
        return prediction.squeeze()
        
        
class MatrixFactorizationPredictor(BasePredictor):
    
    def __init__(self,
                 model_list,
                 key,
                 dim=128,
                 text_dim=768,
                 offline_lr=0.001,
                 offline_epoch=1,
                 online_lr=0.01,
                 batch_size=32,
                 online_decay=0.99,
                 SFT_dataset=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        self.model = MatrixFactorization(
            num_models=len(model_list),
            dim=dim,
            text_dim=text_dim    
        ).to(device)
        self.model_list = model_list
        self.batch_size = batch_size
        self.lr = online_lr
        self.decay = online_decay
        self.device = device
        self.key = key
        self.loss = nn.MSELoss(reduction="mean")
        if SFT_dataset is not None:
            self.offline_training(SFT_dataset, key=key, lr=offline_lr,epoch=offline_epoch)
        self.buffer = []
    
    def sample2input(self, sample):
         """
         sample must be a dictionary which contains the prompt_embedding and model_id key
         """
         return sample["prompt_embedding"], sample["model_name"], sample[self.key]
            
    
    def offline_training(self, dataset:SFTDataset, key:str, lr:float=0.001, epoch=1):
        sft_dataloader = SFTDataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epoch):
            for batch in sft_dataloader:
                x = torch.tensor([sample["prompt_embedding"] for sample in batch], dtype=torch.float32).to(self.device)
                a = torch.tensor([self.model_list.index(sample["model_name"]) for sample in batch], dtype=torch.long).to(self.device)
                y = torch.tensor([sample[key] for sample in batch], dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                prediction = self.model(x, a)
                loss = self.loss(prediction, y)
                loss.backward()
                optimizer.step()
                
    def online_update(self, sample):
        x, a, y = self.sample2input(sample)
        if len(self.buffer) < self.batch_size:
           self.buffer.append((x, a, y))
        else:
            x_batch, a_batch, y_batch = zip(*self.buffer)
            x_batch = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
            a_batch = torch.tensor(a_batch, dtype=torch.long).to(self.device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(self.device)
            self.model.train()
            prediction = self.model(x_batch, a_batch)
            loss = self.loss(prediction, y_batch)
            loss.backward(self.lr)
            self.lr *= self.decay
            self.buffer = []
    
    def predict(self, model_id, prompt_embedding):
        self.model.eval()
        return self.model.forward(model_id, prompt_embedding).detach().numpy()



if __name__=="__main__":
    model = XGB()
    model.predict(np.zeros((11,768+768)))
