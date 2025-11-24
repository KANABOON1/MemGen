from abc import ABC, abstractmethod
from typing import Literal, Dict, Tuple

class BaseEnv(ABC):
    ENV_CARD: Literal["STATIC", "DYNAMIC"] = None

    def __init__(self, config):
        self.config = config

    @classmethod
    @abstractmethod
    def compute_reward(cls, **kwargs):
        ...


class StaticEnv(BaseEnv):
    ENV_CARD = "STATIC"
    

class DynamicEnv(BaseEnv):
    ENV_CARD = "DYNAMIC"
    
    @abstractmethod
    def set_env(self, task_config: Dict) -> Tuple[str, str]:  
        ...

    @classmethod
    @abstractmethod
    def preprocess_action(self, action: str) -> str:
        ...
    
    @abstractmethod
    def step(self, action: str) -> Tuple[str, bool]:
        ...
    
    @abstractmethod
    def feedback(self) -> Tuple[float, bool]:
        ...