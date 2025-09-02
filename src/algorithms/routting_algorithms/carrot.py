from src.algorithms.routting_algorithms.base_model import OnlineModel
from src.algorithms.predictor.base_model import BasePredictor
from src.logger import Logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class CarrotRouter(OnlineModel):
    def __init__(self, budget):
        self.budget = budget
        self.t = 0
        # Define model costs
        self.COSTS = {
            'aws-claude-3-5-sonnet-v1': [3, 15], 'aws-titan-text-premier-v1': [0.5, 1.5],
            'openai-gpt-4o': [2.5, 10], 'openai-gpt-4o-mini': [0.15, 0.6],
            'wxai-granite-3-2b-instruct-8k-max-tokens': [0.1, 0.1],
            'wxai-granite-3-8b-instruct-8k-max-tokens': [0.2, 0.2],
            'wxai-llama-3-1-70b-instruct': [0.9, 0.9], 'wxai-llama-3-1-8b-instruct': [0.2, 0.2],
            'wxai-llama-3-2-1b-instruct': [0.06, 0.06], 'wxai-llama-3-2-3b-instruct': [0.06, 0.06],
            'wxai-llama-3-3-70b-instruct': [0.9, 0.9], 'wxai-mixtral-8x7b-instruct-v01': [0.6, 0.6],
            'wxai-llama-3-405b-instruct': [3.5, 3.5]
        }
        self.mu = 0.2

        # Load tokenizers and models
        self.input_counter = AutoTokenizer.from_pretrained("./models/Llama-3.2-3B")
        self.tokenizer = AutoTokenizer.from_pretrained('./models/roberta-base')

        self.score_predictor = AutoModelForSequenceClassification.from_pretrained(
            './models/carrot-preformance',
            problem_type="multi_label_classification",
            num_labels=len(self.COSTS),
        )

        self.output_counter = AutoModelForSequenceClassification.from_pretrained(
            './models/carrot-cost',
            problem_type="regression",
            num_labels=len(self.COSTS),
        )

        # Map index to model names
        self.id2label = list(self.COSTS.keys())

    def take_action(self, sample):
        self.current_sample = sample.copy()
        prompt = sample["prompt"]
        tokenized_text = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            is_split_into_words=False,
            return_tensors='pt'
        )

        self.input_counter.pad_token = self.tokenizer.eos_token

        scores = 1 / (1 + np.exp(-self.score_predictor(tokenized_text["input_ids"]).logits.detach().numpy()))
        output_tokens = self.output_counter(tokenized_text["input_ids"]).logits.detach().numpy().T
        input_tokens = self.input_counter(prompt, return_tensors="pt")["input_ids"].shape[1]
        input_tokens = np.array(input_tokens).T

        costs = []
        for i, model in enumerate(self.COSTS.keys()):
            cost = (input_tokens * self.COSTS[model][0] / 1_000_000) + (output_tokens[i] * self.COSTS[model][1] / 1_000)
            costs.append(cost.tolist())

        costs = np.array(costs).T
        model_idx = ((1 - self.mu) * scores - self.mu * costs * 100).argmax(axis=1, keepdims=True)
        action = [self.id2label[idx[0]] for idx in model_idx][0]
        self.current_sample["model_name"] = action
        self.current_sample["model_index"] = model_idx[0][0]
        return action
        
    
    def update(self, reward, cost, response):
        self.current_sample["reward"] = reward
        self.current_sample["cost"] = cost
        ret_sample = self.current_sample.copy()
        ret_sample["response"] = response
        self.current_sample = None
        self.budget -= cost
        self.t += 1
        return ret_sample