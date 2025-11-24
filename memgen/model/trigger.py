from peft import LoraConfig, PeftModel
import torch
import torch.nn as nn


class MemGenTrigger(nn.Module):
    adapter_name = "trigger"

    def __init__(
        self,
        model: PeftModel,
        active: bool,
    ):
        super().__init__()
        
        self.active = active
        self.model = model
        self.output_layer = nn.Linear(model.base_model.config.hidden_size, 2)

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor,
        position_ids: torch.Tensor
    ) -> torch.FloatTensor:
        
        if self.active:
            self.model.set_adapter(self.adapter_name)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,  
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            logits = self.output_layer(hidden_states)
            self.model.disable_adapter()
        
        else:
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)  # logits: [batch_size, seq_len, 2]
            logits[..., 1] = 1.0  

        return logits



