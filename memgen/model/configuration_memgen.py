from transformers import PretrainedConfig
from typing import Optional


class MemGenConfig(PretrainedConfig):
    model_type = "memgen"

    def __init__(
        self,
        # weaver configs
        weaver_lora_config: Optional[dict] = None,
        prompt_latents_len: int = 0,
        inference_latents_len: int = 0,
        # trigger configs
        trigger_active: bool = False,
        trigger_lora_config: Optional[dict] = None,
        max_prompt_aug_num: int = 1,
        max_inference_aug_num: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # weaver configs
        self.weaver_lora_config = weaver_lora_config
        self.prompt_latents_len = prompt_latents_len
        self.inference_latents_len = inference_latents_len

        # trigger configs
        self.trigger_active = trigger_active
        self.trigger_lora_config = trigger_lora_config
        self.max_prompt_aug_num = max_prompt_aug_num
        self.max_inference_aug_num = max_inference_aug_num
