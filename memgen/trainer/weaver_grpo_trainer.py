import copy
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational 
from trl.models import create_reference_model, prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
if is_peft_available():
    from peft import PeftConfig, get_peft_model
if is_wandb_available():
    import wandb

from interactions.base_interaction import (
    InteractionManager, InteractionDataProto
)
from data.base_env import StaticEnv, DynamicEnv

from .utils import (
    nanstd, nanmax, nanmin
)
from ..model.modeling_memgen import MemGenModel

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class WeaverGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        model: MemGenModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        env_class = None,   # env main class
        env_main_config = None,  # configs to initialize an env object
        generation_manager: InteractionManager = None  # manage the interaction between agent and env
    ):
        super().__init__(
            model, 
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config
        )
        
        self.env_class = env_class
        self.env_main_config = env_main_config
        self.generation_manager = generation_manager
        
        assert self.max_prompt_length == generation_manager.config.max_start_length
        assert self.max_completion_length == generation_manager.config.max_response_length
        assert self.temperature == generation_manager.config.temperature   
    
    def _build_multiturn_envs(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> tuple[list[list[dict]], list]:
        init_messages, envs = [], []

        for task_config in inputs:
            env: DynamicEnv = self.env_class(self.env_main_config)
            system_prompt, init_user_prompt = env.set_env(task_config)
            
            system_message = {"role": "system", "content": system_prompt}
            init_user_message = {"role": "user", "content": init_user_prompt}
            
            init_messages.append([system_message, init_user_message])
            envs.append(env)
        
        return init_messages, envs
    
    def _get_per_token_logps(
        self, model, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: torch.Tensor, 
        logits_to_keep: int,
        batch_size: int = None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        supervise_masks = []   
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels}

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            outputs = model(**model_inputs)
            logits = outputs.logits
            labels = outputs.supervised_labels
            
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)
            
            labels = labels[:, -logits_to_keep:]
            mask = (labels != -100).long()  
            supervise_masks.append(mask)

        logps = torch.cat(all_logps, dim=0)
        masks = torch.cat(supervise_masks, dim=0)
        return logps, masks


    # NOTE - currently we only deal with text input and leave multimodal as a feature work
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]  # batch_size * num_generations
    ) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # build no-tensor part
        batch_gen_keys = []
        if "prompt" in inputs[0]:  # text-based raw prompt
            batch_gen_keys.append("prompt")
        if "tools_kwargs" in inputs[0]:  # tool-integrated     
            batch_gen_keys.append("tools_kwargs")
        if "interaction_kwargs" in inputs[0]:  # interaction args
            batch_gen_keys.append("interaction_kwargs")
        if "agent_name" in inputs[0]:  # agent name
            batch_gen_keys.append("agent_name")    

        gen_batch = InteractionDataProto()
        for key in batch_gen_keys:  
            gen_batch.no_tensor_batch[key] = [x[key] for x in inputs]
        
        # Single-turn env
        if issubclass(self.env_class, StaticEnv):
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
                
            prompts, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
            if self.max_prompt_length is not None:
                prompts = prompts[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

            gen_batch.batch["input_ids"] = prompts 
            gen_batch.batch["attention_mask"] = prompt_mask
        # Multi-turn env
        elif issubclass(self.env_class, DynamicEnv):
            init_prompts, envs = self._build_multiturn_envs(inputs)
            gen_batch.no_tensor_batch["init_prompts"] = init_prompts
            gen_batch.no_tensor_batch["envs"] = envs
            
            for example, env in zip(inputs, envs):
                example["envs"] = env
        else:
            raise ValueError("Unsupported environment type")
        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                sampling_params = SamplingParams(
                    n=1,  # vLLM on each GPU generates only 1 in colocate mode
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                )

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompts, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    # Use GenerationManager to coordinate the interaction between the agent and the environment
                    self.generation_manager.actor_rollout_wg = unwrapped_model  
                    final_gen_batch_output = self.generation_manager.run_agent_loop(gen_batch=gen_batch)
        
        # parse outputs
        prompts = final_gen_batch_output.batch["prompts"].to(device)  # prompt ids
        completion_ids = final_gen_batch_output.batch["responses"].to(device)  # completion ids
        prompt_completion_ids = final_gen_batch_output.batch["input_ids"].to(device)  # prompt and completion ids
        attention_mask = final_gen_batch_output.batch["attention_mask"].to(device)  # attention_mask on prompt and response
        prompt_mask = attention_mask[:, :prompts.size(1)]  
        completion_mask = final_gen_batch_output.batch["info_mask"][:, prompts.size(1):].to(device) 
        is_eos = completion_ids == self.eos_token_id
        assert completion_ids.shape == completion_mask.shape

        # Construct labels: Supervise only the agent response portion.
        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        
        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        logits_to_keep = completion_mask.size(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        with torch.no_grad():
            # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            # old_per_token_logps == per_token_logps, so we can skip it's computation here, and use
            # per_token_logps.detach() instead.
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps, old_supervise_mask = self._get_per_token_logps( 
                    self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                )
            else:
                old_per_token_logps, old_supervise_mask = None, None
            
            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                        )
            else: 
                ref_per_token_logps, ref_supervise_mask = None, None
        
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text
        
        # compute rewards
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())
        
        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        # self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompts,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "old_supervise_mask": old_supervise_mask,   
            "ref_per_token_logps": ref_per_token_logps,
            "ref_supervise_mask": ref_supervise_mask
        }


    def _compute_loss(self, model, inputs):
        device = self.accelerator.device

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        old_supervise_mask, ref_supervise_mask = inputs["old_supervise_mask"], inputs["ref_supervise_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        logits_to_keep = completion_labels.size(1)
        
        assert prompt_ids.shape == prompt_mask.shape
        assert completion_ids.shape == completion_mask.shape
        assert input_ids.shape == attention_mask.shape == labels.shape
        per_token_logps, supervise_mask = self._get_per_token_logps(model, input_ids, attention_mask, labels, logits_to_keep)
        
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        if old_supervise_mask is None:
            old_supervise_mask = supervise_mask
        if ref_supervise_mask is None:
            ref_supervise_mask = supervise_mask
        # Consistency check: The positions that are supervised must be a subset of the completion mask.
        assert (
            torch.all(supervise_mask <= completion_mask) and
            torch.all(old_supervise_mask <= completion_mask) and
            torch.all(ref_supervise_mask <= completion_mask)
        )
        supervised_mask = completion_mask * supervise_mask * old_supervise_mask * ref_supervise_mask  

        if self.loss_type == "grpo":
            loss = ((per_token_loss * supervised_mask).sum(-1) / supervised_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * supervised_mask).sum() / supervised_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * supervised_mask).sum() / (supervised_mask.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * supervised_mask).sum() / supervised_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * supervised_mask).sum() / supervised_mask.sum()
        high_clip = (is_high_clipped * supervised_mask).sum() / supervised_mask.sum()
        clip_ratio = (is_region_clipped * supervised_mask).sum() / supervised_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss