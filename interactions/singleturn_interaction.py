import torch
from typing import Dict, List
from transformers import GenerationConfig

from interactions.base_interaction import (
    InteractionConfig, 
    InteractionManager,
    InteractionDataProto
)


class SingleTurnInteractionManager(InteractionManager):
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        super().__init__(
            tokenizer, actor_rollout_wg, config, is_validation
        )
        # generation configs for agent
        self.generation_config = GenerationConfig(
            do_sample=self.config.do_sample,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(  
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']    

    def _info_masked_concatenate_with_padding(self, 
        prompt: torch.Tensor, 
        prompt_with_mask: torch.Tensor, 
        response: torch.Tensor, 
        info: torch.Tensor = None,
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info
    
    def _update_right_side(
        self, right_side: Dict, 
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None
    ) -> Dict:
        """Update right side state."""
        if next_obs_ids != None: 
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False   
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}
    
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        
        initial_input_ids = gen_batch.batch["input_ids"]
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        # postprocess model inputs
        rollings = gen_batch
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask']    
        )
        rollings_active = {
            k: v for k, v in rollings.batch.items()
        }  

        # model generation
        gen_output = self.actor_rollout_wg.generate(
            rollings_active["input_ids"], 
            rollings_active["attention_mask"], 
            generation_config=self.generation_config
        )
        responses_ids = gen_output[:, rollings_active["input_ids"].size(1):]
        responses_ids = self.tensor_fn.erase_after_first_eos(responses_ids, self.tokenizer.eos_token_id)
        
        # update right side
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids=None)
        
        # construct final output
        return self._compose_final_output(original_left_side, original_right_side)
    
    def _compose_final_output(
        self, left_side: Dict,
        right_side: Dict,
    ) -> InteractionDataProto:
        """Compose final generation output."""
        
        final_output_batch = right_side.copy()
        final_output_batch['prompts'] = left_side['input_ids']
        final_output_batch["responses"] = right_side['responses']
        
        # Combine input IDs: input_ids + responses
        final_output_batch['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask
        final_output_batch['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output_batch['responses'])
        ], dim=1)
        
        final_output_batch['info_mask'] = torch.cat([  
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output_batch['responses_with_info_mask'])
        ], dim=1)
        
        final_output = InteractionDataProto(batch=final_output_batch)

        return final_output
        