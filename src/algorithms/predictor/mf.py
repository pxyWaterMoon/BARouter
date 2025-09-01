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
                 key,
                 ):
        super(MatrixFactorization, self).__init__()
        self.model_embed = nn.Embedding(num_models, dim)
        self.text_projection = nn.Linear(text_dim, dim, bias=False)
        self.classifier = nn.Linear(dim, 1, bias=False)  # Assuming binary classification for reward/cost prediction
        self.key = key

    def forward(self, model_id, prompt_embedding):
        model_embedding = self.model_embed(model_id)
        text_embedding = self.text_projection(prompt_embedding)
        prediction = self.classifier(model_embedding * text_embedding)

        ## debug 
        # return torch.sigmoid(prediction).squeeze()
        if self.key == "reward":
            return torch.sigmoid(prediction).squeeze()
        elif self.key == "cost":
            return torch.relu(prediction).squeeze()
        else:
            raise ValueError("The key must be either 'reward' or 'cost'.")
        return prediction.squeeze()

class MatrixFactorizationPredictor(BasePredictor):
    
    def __init__(self,
                 model_list,
                 key,
                 dim=128,
                 text_dim=75,   #768,75
                 offline_lr=0.01,
                 offline_epoch=1,
                 online_lr=0.01,
                 buffer_size=64,
                 online_decay=0.99,
                 SFT_dataset=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 logger=None,
                 ):
        self.model = MatrixFactorization(
            num_models=len(model_list),
            dim=dim,
            text_dim=text_dim,
            key=key
        ).to(device)
        self.model_list = model_list
        self.buffer_size = buffer_size
        self.lr = online_lr
        self.decay = online_decay
        self.device = device
        self.key = key
        self.loss = nn.MSELoss(reduction="mean")
        self.logger = logger
        if SFT_dataset is not None:
            self.offline_training(SFT_dataset, key=key, lr=offline_lr,epoch=offline_epoch)
        self.buffer = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=online_lr)
        
    
    def sample2input(self, sample_list:list[dict], key = None):
        if all("prompt_embedding" not in sample for sample in sample_list) or all("model_index" not in sample for sample in sample_list):
             raise ValueError("The input sample_list must contain 'prompt_embedding' and 'model_index' keys for Matrix Factorization predictor.")
        x = np.array([sample["prompt_embedding"] for sample in sample_list])
        a = np.array([sample["model_index"] for sample in sample_list])
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        if key is not None:
            y = np.array([sample[key] for sample in sample_list])
            y = torch.tensor(y, dtype=torch.float32).to(self.device)
            return x, a, y
        return x, a
            
    
    def offline_training(self, dataset:SFTDataset, key:str, lr:float=0.001, epoch=1):
        sft_dataloader = SFTDataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        step = 0
        for _ in range(epoch):
            for batch in sft_dataloader:
                x = torch.tensor(np.array([sample["prompt_embedding"] for sample in batch]), dtype=torch.float32).to(self.device)
                a = torch.tensor(np.array([self.model_list.index(sample["model_name"]) for sample in batch]), dtype=torch.long).to(self.device)
                y = torch.tensor(np.array([sample[key] for sample in batch]), dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                prediction = self.model(a, x)
                loss = self.loss(prediction, y)
                loss.backward()
                self.logger.log_scalar(
                    {
                        f"predictor/{key}_offline_loss": loss.item(),
                        f"predictor/{key}_offline_lr": optimizer.param_groups[0]['lr'],
                     },
                    step=step)
                optimizer.step()
                step += 1
        print(f"Successfully trained the predictor of {key}.")
                
    def online_update(self, sample, global_step):
        self.buffer.append(sample)
        if len(self.buffer) >= self.buffer_size:
            x, a, y = self.sample2input(self.buffer, key=self.key)
            self.model.train()
            online_epoch = 1
            for _ in range(online_epoch):
                self.optimizer.zero_grad()
                prediction = self.model(a, x)
                loss = self.loss(prediction, y)
                loss.backward()
                self.logger.log_scalar(
                    {
                        f"predictor/{self.key}_online_loss": loss.item(),
                        f"predictor/{self.key}_online_lr": self.optimizer.param_groups[0]['lr'],
                    },
                    step=global_step
                )
                self.optimizer.step()
            self.buffer = []


        # x, a, y = self.sample2input(sample)
        # if len(self.buffer) < self.batch_size:
        #    self.buffer.append((x, a, y))
        # else:
        #     x_batch, a_batch, y_batch = zip(*self.buffer)
        #     x_batch = torch.tensor(x_batch, dtype=torch.float32).to(self.device)
        #     a_batch = torch.tensor(a_batch, dtype=torch.long).to(self.device)
        #     y_batch = torch.tensor(y_batch, dtype=torch.float32).to(self.device)
        #     self.model.train()
        #     prediction = self.model(x_batch, a_batch)
        #     loss = self.loss(prediction, y_batch)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     self.buffer = []
    
    def predict(self, sample_list):
        self.model.eval()
        x, a = self.sample2input(sample_list)
        return self.model.forward(a, x).detach().cpu().numpy()

