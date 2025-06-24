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
    model = SentenceTransformer(model_path, device=device)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding batches"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False).detach().cpu()
        embeddings.append(batch_embeddings)    
    return torch.cat(embeddings, dim=0)


