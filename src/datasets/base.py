from torch.utils.data import Dataset
from typing_extensions import TypedDict  # Python 3.10+
from typing_extensions import NotRequired  # Python 3.11+
from typing import Any, Callable, ClassVar, Collection, Dict, Iterable, Iterator
from weakref import WeakValueDictionary
import abc

__all__ = [
    'RawDataset',
    'RawSample',
]

class RawSample(TypedDict, total=False):
    """Raw sample type.

    For each sample, should provide prompt, model description.
    If the embedding of the prompt is available, it can be provided as well.
    For supervised datasets, should provide model name and the corresponding response, reward and cost.
    For online updating datasets, should some available method to get the reward and cost.
        1. If there is a ground turth can be directly checked, it should be provided as well. The ground truth should be a dictionary with model names as keys and the values are also a dictionary with response, reward and cost as keys.
    """
    # prompt: str
    prompt: NotRequired[str]
    prompt_embedding: NotRequired[list[float]]
    available_models_description: NotRequired[dict[str, str]]
    available_models_description_embeddings: NotRequired[dict[str, list[float]]]

    # For supervised datasets
    model_name: NotRequired[str]
    reward: NotRequired[float]
    cost: NotRequired[float]

    # For online updating datasets
    ground_truth: NotRequired[dict[str, dict[str, float|str]]]

class RawDataset(Dataset[RawSample]):
    """ Base class for raw datasets."""

    NAME: ClassVar[str]
    """ Name of the dataset. """
    __REGISTRY: ClassVar[WeakValueDictionary[str, RawDataset]] = WeakValueDictionary()
    """Registry of all subclasses."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        """Register the subclass."""
        name = getattr(cls, 'NAME', None)
        if name is None:
            return
        if name in cls.__REGISTRY:
            raise ValueError(f"Dataset {name} is already registered.")
        cls.__REGISTRY[name] = cls
    
    @staticmethod
    def load(name: str) -> 'RawDataset':
        """Load a dataset by name."""
        try:
            cls = RawDataset.__REGISTRY[name]
        except KeyError:
            raise ValueError(f"Dataset {name} is not registered.")
        return cls()
    
    @abc.abstractmethod
    def __getitem__(self, index) -> RawSample:
        raise NotImplementedError("Subclasses must implement __getitem__ method.")
    
    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__ method.")
    
    def __iter__(self) -> Iterator[RawSample]:
        """Return an iterator over the dataset."""
        return (self[i] for i in range(len(self))
        )
    
    
