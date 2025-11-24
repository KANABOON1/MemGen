from abc import ABC, abstractmethod
from typing import Type

from datasets import DatasetDict

from data.base_env import BaseEnv

class BaseBuilder(ABC):

    def __init__(self, cfg: dict = None):
        super().__init__()
        
        self.mode = cfg.get("mode", "sft")
        self.config = cfg.get(self.mode)
    
    def get_dataset_dict(self) -> DatasetDict:
        method_builder_map = {
            "sft": self._build_sft_datasets,
            "grpo": self._build_rl_datasets,
        }

        if self.mode not in method_builder_map:
            raise ValueError("Unsupported datasets mode")
        
        return method_builder_map[self.mode]()
    
    @abstractmethod
    def get_env_cls(self) -> Type[BaseEnv]:
        ...

    @abstractmethod
    def _build_sft_datasets(self) -> DatasetDict:
        ...
    
    @abstractmethod
    def _build_rl_datasets(self) -> DatasetDict:
        ...
    
