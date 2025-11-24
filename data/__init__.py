from data.base_builder import BaseBuilder
from data.base_env import (
    BaseEnv,
    StaticEnv,
    DynamicEnv,
)
from data.gpqa.builder import GPQABuilder
from data.gsm8k.builder import GSM8KBuilder
from data.kodcode.builder import KodCodeBuilder
from data.triviaqa.builder import TriviaQABuilder

_DATA_BUILDER_MAP = {
    "gpqa": GPQABuilder,
    "gsm8k": GSM8KBuilder,
    "kodcode": KodCodeBuilder,
    "triviaqa": TriviaQABuilder
}

def get_data_builder(dataset_cfg) -> BaseBuilder:
    if dataset_cfg.get("name") not in _DATA_BUILDER_MAP:
        raise ValueError("Unsupported dataset.")
    
    builder_cls = _DATA_BUILDER_MAP[dataset_cfg.get("name")]
    builder = builder_cls(dataset_cfg)

    return builder