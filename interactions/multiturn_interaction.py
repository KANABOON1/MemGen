import torch
from typing import Dict, List, Tuple
from transformers import GenerationConfig
import copy

from interactions.base_interaction import (
    InteractionDataProto,
    InteractionConfig, 
    InteractionManager
)


class MultiTurnInteractionManager(InteractionManager):
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
    
    def _build_chat_history(self, rollings: Dict) -> List[Dict]:

        init_prompts = rollings.get("init_prompts")
        if init_prompts is None:
            raise ValueError("")
        
        inter_histories = rollings.get("inter_histories")
        if inter_histories is None:
            raise ValueError("")
        
        chat_histories: List[List[Dict]] = []
        for init_prompt, inter_history in zip(init_prompts, inter_histories):
            chat_histories.append(init_prompt + inter_history)

        return chat_histories
    
    def _update_interaction_history(self, rollings: InteractionDataProto, responses: List[str], observations: List[str]) -> List[List[Dict]]:

        inter_histories = copy.deepcopy(rollings.no_tensor_batch.get("inter_histories"))
        assert len(inter_histories) == len(responses) == len(observations)
        for inter_history, response, observation in zip(inter_histories, responses, observations):
            assistant_info = {"role": "assistant", "content": response}
            user_info = {"role": "user", "content": observation}
            
            inter_history.append(assistant_info)
            inter_history.append(user_info)
        
        return inter_histories
    
    def _postprocess_responses(self, responses: torch.Tensor, envs: List) -> torch.Tensor:

        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        processed_responses_str = []
        for r, env in zip(responses_str, envs):
            processed_r = env.preprocess_action(r)
            processed_responses_str.append(processed_r)

        responses = self._batch_tokenize(processed_responses_str)
        return responses, processed_responses_str


    def _example_level_pad(
        self, responses_ids: torch.Tensor, responses_str: List[str], active_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:

        assert active_mask.sum() == responses_ids.shape[0]
        # Create masked responses tensor
        batch_size = active_mask.shape[0]
        seq_len = responses_ids.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.tokenizer.pad_token_id,
            dtype=responses_ids.dtype, device=responses_ids.device
        )
        padded_responses[active_mask] = responses_ids   
        
        # Create masked response strings
        padded_responses_str = [""] * batch_size
        
        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1
                
        return padded_responses, padded_responses_str

    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """Run main LLM generation loop (conversation format)."""
        assert "init_prompts" in gen_batch.no_tensor_batch
        assert "envs" in gen_batch.no_tensor_batch
        batch_size = len(gen_batch.no_tensor_batch["init_prompts"])

        rollings = gen_batch   
        rollings.no_tensor_batch["inter_histories"] = [[] for _ in range(batch_size)]

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]

        for step in range(self.config.max_turns):
            if not active_mask.sum():   
                break            

            mask_list = active_mask.tolist()  
            rollings_active = {
                k: [item for item, keep in zip(v, mask_list) if keep]
                for k, v in rollings.no_tensor_batch.items()
            }
            # use tokenizer to add chat template and encode text to tokens: input_ids, attention_mask
            messages = self._build_chat_history(rollings_active)
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, 
                add_generation_prompt=True, 
                padding=True, return_tensors="pt", return_dict=True
            )

            # agent rollout
            gen_output = self.actor_rollout_wg.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=self.generation_config,
            ).to("cpu")

            # postprocess
            prompt_len = inputs["input_ids"].size(1)
            responses = gen_output[:, prompt_len:]
            responses = self.tensor_fn.erase_after_first_eos(responses, self.tokenizer.eos_token_id)
            responses_ids, responses_str = self._postprocess_responses(responses, rollings_active["envs"])
            all_responses_ids, all_responses_str = self._example_level_pad(responses_ids, responses_str, active_mask)

            next_obs, dones = self._execute_predictions(rollings, all_responses_str, active_mask)
            processed_obs = self._postprocess_observations(next_obs)
            
            # post process interaction states
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())

            interaction_histories = self._update_interaction_history(rollings, all_responses_str, processed_obs)
            rollings.no_tensor_batch["inter_histories"] = interaction_histories
  
        # build final outputs
        final_outputs = self._build_final_outputs(rollings)
        return final_outputs

    def _execute_predictions(self, rollings: InteractionDataProto, responses: List[str], active_mask: torch.Tensor) -> Tuple[List[str], List[str]]:
        observations = []
        dones = []
        for response, env, is_active in zip(responses, rollings.no_tensor_batch["envs"], active_mask):
            if is_active:
                observation, _, done = env.step(response)
            else:   
                observation = ""
                done = True
            observations.append(observation)
            dones.append(done)

        return observations, dones

    
    def _postprocess_observations(self, observations: List[str]) -> List[str]:
        self.tokenizer.padding_side = "right" 
        next_obs_ids = self._batch_tokenize(observations)

        max_len = self.config.max_obs_length
        if next_obs_ids.shape[1] > max_len:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {max_len}")

            extra_text = "..."
            extra_ids = self.tokenizer.encode(
                extra_text, add_special_tokens=False, return_tensors="pt"
            ).to(next_obs_ids.device)
            extra_len = extra_ids.shape[1]

            new_obs_ids = []
            for row in next_obs_ids:
                valid_len = (row != self.tokenizer.pad_token_id).sum().item()

                if valid_len > max_len:  
                    truncated = row[: max_len - extra_len]
                    new_row = torch.cat([truncated, extra_ids.squeeze(0)], dim=0)
                else:
                    new_row = row[:max_len]

                new_obs_ids.append(new_row.unsqueeze(0))

            next_obs_ids = torch.cat(new_obs_ids, dim=0)
            observations = self.tokenizer.batch_decode(next_obs_ids, skip_special_tokens=True)

        return observations

    def _build_final_outputs(self, rollings: InteractionDataProto) -> InteractionDataProto:

        init_prompts: List[List[Dict]] = rollings.no_tensor_batch["init_prompts"]
        inter_histories: List[List[Dict]] = rollings.no_tensor_batch["inter_histories"]
        
        output = InteractionDataProto()

        output.no_tensor_batch["inter_histories"] = [
            prompt + inter for prompt, inter in zip(init_prompts, inter_histories)
        ]
        
        # ---------- prompts ----------
        self.tokenizer.padding_side = "left"
        prompt_ids = self.tokenizer.apply_chat_template(                
            init_prompts, tokenize=True, 
            add_generation_prompt=False,  
            padding=True, return_tensors="pt", return_dict=True
        )
        output.batch["prompts"] = prompt_ids["input_ids"]
        prompt_attn_mask = prompt_ids["attention_mask"]
        
        # ---------- responses ----------
        self.tokenizer.padding_side = "right"
        response_ids = self.tokenizer.apply_chat_template(                
            inter_histories, 
            tokenize=True, 
            padding=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_tensors="pt", return_dict=True
        )
        output.batch["responses"] = response_ids["input_ids"]
        response_attn_mask = response_ids["attention_mask"]

        completion_info_mask = response_ids["assistant_masks"]

        # ---------- input_ids ----------
        output.batch["input_ids"] = torch.cat(
            [prompt_ids["input_ids"], response_ids["input_ids"]], dim=1
        )
        output.batch["attention_mask"] = torch.cat(
            [prompt_attn_mask, response_attn_mask], dim=1
        )

        # ---------- info_mask ----------
        prompt_info_mask = torch.zeros(
            prompt_ids["input_ids"].shape, 
            dtype=completion_info_mask.dtype, 
            device=completion_info_mask.device
        )

        output.batch["info_mask"] = torch.cat(
            [prompt_info_mask, completion_info_mask], dim=1
        )
        
        self.tokenizer.padding_side = "left"

        return output