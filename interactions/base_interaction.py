from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from interactions.tensor_utils import TensorHelper, TensorConfig


@dataclass
class InteractionConfig:
    max_turns: int = 1
    max_start_length: int = 1024   
    max_prompt_length: int = 4096   
    max_response_length: int = 512
    max_obs_length: int = 512
    do_sample: bool = False
    temperature: float = 1.0  
    batch_size: int = 8
    output_dir: Optional[str] = None

@dataclass
class InteractionDataProto:
    batch: dict = field(default_factory=dict)
    no_tensor_batch: dict = field(default_factory=dict)

class InteractionManager(ABC):
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" 
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        assert tokenizer.pad_token_id is not None
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
    
    @abstractmethod
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        ...