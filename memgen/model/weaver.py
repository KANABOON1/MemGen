from peft import LoraConfig, PeftModel
import torch
import torch.nn as nn


class MemGenWeaver(nn.Module):

    adapter_name = "weaver"

    def __init__(
        self, 
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
    ):
        super().__init__()

        self.model = model

        # prompt augmentation
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, model.base_model.config.hidden_size), 
            requires_grad=True
        )

        # inference augmentation
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, model.base_model.config.hidden_size), 
            requires_grad=True
        )
    
    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _augment(
        self, 
        latents: torch.Tensor,
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        self.model.set_adapter(self.adapter_name)

        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # inputs_embeds
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total)
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        
        # get position ids
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1) 

        # the processor only outputs the hidden states
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        self.model.disable_adapter()  

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.prompt_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )


    def augment_inference(
        self, 
        inputs_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.inference_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )