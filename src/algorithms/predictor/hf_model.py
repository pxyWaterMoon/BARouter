import pandas as pd
import numpy as np
from src.algorithms.predictor.base_model import BasePredictor
from src.algorithms.utils import embedding_batch
import torch.nn as nn
import torch
from src.datasets.sftdata import SFTDataset, SFTDataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HFMoodelPredictor(BasePredictor):
    
    def __init__(self,
                 model_list,
                 key,
                 model_name_or_path,
                 cost_table = None,
                 input_counter_path_or_name = None,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_list = model_list
        self.key = key
        self.predictor = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            problem_type="multi_label_classification" if key == "reward" else "regression",
            num_labels=len(model_list),
        )
        if self.key == "length":
            self.cost_table = cost_table
            input_counter_path_or_name = "./models/Llama-3.2-3B" if input_counter_path_or_name is None else input_counter_path_or_name
            self.input_counter = AutoTokenizer.from_pretrained(input_counter_path_or_name)
        

    def predict(self, sample_list):
        prompt = sample_list[0]["prompt"]
        tokenized_text = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            is_split_into_words=False,
            return_tensors='pt'
        )
        if self.key == "reward":
            scores = 1 / (1 + np.exp(-self.predictor(tokenized_text["input_ids"]).logits.detach().numpy()))

            #### orders between [output_tokens] and [self.model_list] are different, need to swap
            tmp = scores[0][-1].copy()
            scores[0][-1] = scores[0][-2]
            scores[0][-2] = tmp

        elif self.key == "length":
            output_tokens = self.predictor(tokenized_text["input_ids"]).logits.detach().numpy().T

            #### orders between [output_tokens] and [self.model_list] are different, need to swap
            tmp = output_tokens[-1].copy()
            output_tokens[-1] = output_tokens[-2]
            output_tokens[-2] = tmp

            input_tokens = self.input_counter(prompt, return_tensors="pt")["input_ids"].shape[1]
            input_tokens = np.array(input_tokens).T
            costs = []
            for i, model in enumerate(self.model_list):
                cost = ((input_tokens * self.cost_table[model][0] / 1_000_000) + (output_tokens[i] * self.cost_table[model][1] / 1_000))*1000
                costs.append(cost.tolist())
            scores = np.array(costs).T
        elif self.key == "cost":
            scores = self.predictor(tokenized_text["input_ids"]).logits.detach().numpy()
        return scores[0]
        
            
    
    def offline_training(self, dataset:SFTDataset, key:str, lr:float=0.001, epoch=1):
        return
                
    def online_update(self, sample, global_step):
        return


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


