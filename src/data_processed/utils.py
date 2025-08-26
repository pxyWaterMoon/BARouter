from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

def text2embeddings(texts, model_path, batch_size=32, device= "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Convert a list of texts into embeddings using a specified SentenceTransformer model.
    Args:
        texts (list): List of strings to convert into embeddings.
        model_path (str): Path to the SentenceTransformer model.
        batch_size (int): Number of texts to process in each batch.
        device (str): Device to run the model on ("cuda" or "cpu").
    Returns:
        torch.Tensor: Tensor of embeddings for the input texts.
    """
    model_name = model_path.split("/")[-1]
    if model_name in ['all-mpnet-base-v2']:
        model = SentenceTransformer(model_path, device=device)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False).detach().cpu()
            embeddings.append(batch_embeddings)    
        return torch.cat(embeddings, dim=0)
    elif model_name in ['bert-base-uncased']:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="embedding batches"):
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Mean Pooling
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).detach().cpu()
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)


