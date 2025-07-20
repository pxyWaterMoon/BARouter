from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

class SentenceTransformerEmbedder:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the embedder with a SentenceTransformer model.
        
        Args:
            model_path (str): Path to the SentenceTransformer model.
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.model = SentenceTransformer(model_path, device=device)

    def embed(self, text):
        """
        Convert a single text into an embedding.
        
        Args:
            text (str): The text to convert into an embedding.
        
        Returns:
            torch.Tensor: The embedding of the input text.
        """
        return self.model.encode(text, convert_to_tensor=True, show_progress_bar=False).detach().cpu()