import os
import random

from accelerate import Accelerator
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig, GRPOConfig
from trl.models import unwrap_model_for_generation

from data import (
    BaseBuilder,
)
from interactions.base_interaction import (
    InteractionConfig,   
    InteractionManager, 
    InteractionDataProto
)
from interactions.singleturn_interaction import SingleTurnInteractionManager
from interactions.multiturn_interaction import MultiTurnInteractionManager

from memgen.model.modeling_memgen import MemGenModel
from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from memgen.trainer.trigger_grpo_trainer import TriggerGRPOTrainer
from memgen.utils import (
    StaticEvalRecorder,
    DynamicEvalRecorder,
    create_tensorboard,
    remove_trainer_checkpoints,
    log_trainable_params,
)

class MemGenRunner:

    def __init__(
        self,
        model: MemGenModel,
        data_builder: BaseBuilder,
        config: dict,
        working_dir: str,
    ):  
        # parse configs
        self.config = config
        self.working_dir = working_dir

        self._parse_configs(config.get("run"))  
        
        # parse model
        self.processing_class = model.tokenizer
        self.model = model

        # initialize envs and generation managers
        self.dataset_dict = data_builder.get_dataset_dict()
        self.env_cls = data_builder.get_env_cls()
        self.env = self.env_cls(config.get("dataset"))

        # partition datasets
        self.weaver_train_dataset, self.trigger_train_dataset = self._parse_train_dataset(self.dataset_dict["train"])
        self.weaver_valid_dataset, self.trigger_valid_dataset = self._parse_valid_dataset(self.dataset_dict["valid"])
        self.test_dataset = self.dataset_dict["test"]
        
        self.weaver_train_dataset = self._filter_dataset(self.weaver_train_dataset)
        self.trigger_train_dataset = self._filter_dataset(self.trigger_train_dataset)
        self.weaver_valid_dataset = self._filter_dataset(self.weaver_valid_dataset)
        self.trigger_valid_dataset = self._filter_dataset(self.trigger_valid_dataset)
        
        # initialize generation manager
        if self.env_cls.ENV_CARD == "STATIC":
            self.inter_cls = SingleTurnInteractionManager
        elif self.env_cls.ENV_CARD == "DYNAMIC":
            self.inter_cls = MultiTurnInteractionManager
        else: 
            raise ValueError("Unsupported environment type.")
        
        self.generation_manager: InteractionManager = self.inter_cls(
            self.processing_class, self.model, self.interaction_config
        )
    
    def _parse_train_dataset(self, train_dataset: Dataset) -> tuple[Dataset, Dataset]:
        
        trigger_trainset_size = min(len(train_dataset) // 2, len(train_dataset))
        rand_indices = random.sample(range(len(train_dataset)), trigger_trainset_size)
        return train_dataset, train_dataset.select(rand_indices)
    
    def _parse_valid_dataset(self, valid_dataset: Dataset) -> tuple[Dataset, Dataset]:

        trigger_validset_size = min(len(valid_dataset) // 2, len(valid_dataset))
        rand_indices = random.sample(range(len(valid_dataset)), trigger_validset_size)
        return valid_dataset, valid_dataset.select(rand_indices)

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class

        # Determine max length based on training mode
        max_len = 1024
        if self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_sft_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_grpo_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_grpo_training_args.max_prompt_length
        else:
            raise ValueError("Wrong training mode.")

        # Function to filter out samples exceeding max length
        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                encoded = tokenizer(sample["prompt"], add_special_tokens=True)
                return len(encoded["input_ids"]) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=True)
                return len(conversation) < max_len
            return True 

        # Apply filtering
        dataset = dataset.filter(filter_func)

        return dataset
    
    # ===== train weaver =====
    def _create_weaver_trainer(self):

        # SFT Trainer
        if self.train_weaver_method == "sft":
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_sft_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
            )
        
        # GRPO Trainer
        elif self.train_weaver_method == 'grpo':
            weaver_trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.weaver_grpo_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
                # --- add env into trainer ---
                env_class=self.env_cls,
                env_main_config=self.config.get("dataset"),
                generation_manager=self.generation_manager
            )
        else:
            raise ValueError("Unsupported weaver training method.")

        return weaver_trainer

    def _train_weaver(self):

        # fix trigger parameters
        self.model.fix_component("trigger")
        self.model.open_component("weaver")
        log_trainable_params(self.model)

        # train weaver
        weaver_trainer = self._create_weaver_trainer()
        weaver_trainer.train()
        weaver_trainer.save_model()   # save the best model
        
        # remove checkpoints and save weaver
        output_dir = weaver_trainer.args.output_dir
        remove_trainer_checkpoints(output_dir)
    
    
    # ===== train trigger =====
    def _create_trigger_trainer(self):
        
        if self.train_trigger_method == "grpo":
            trigger_trainer = TriggerGRPOTrainer(
                model=self.model, 
                processing_class=self.processing_class, 
                train_dataset=self.trigger_train_dataset, 
                eval_dataset=self.trigger_valid_dataset, 
                reward_funcs=[self.env_cls.compute_reward],
                args=self.trigger_grpo_training_args
            )
        else:
            raise ValueError("Unsupported trigger training method.")

        return trigger_trainer
    
    def _train_trigger(self):

        # fix weaver parameters
        self.model.fix_component("weaver")
        self.model.open_component("trigger")
        log_trainable_params(self.model)

        # train trigger
        trigger_trainer = self._create_trigger_trainer()
        trigger_trainer.train()
        trigger_trainer.save_model()     # save the best model

        # remove checkpoints and save weaver
        output_dir = trigger_trainer.args.output_dir
        remove_trainer_checkpoints(output_dir)

    
    # ===== train weaver/trigger =====
    def train(self):
        # train weaver
        if self.train_weaver:
            self._train_weaver()
            
        # train trigger
        if self.train_trigger:
            self._train_trigger()
    
    # ===== evaluate =====
    def evaluate(self):
        self.model = self.model.to(torch.bfloat16)
        self.model.fix_component("weaver")
        self.model.fix_component("trigger")

        evaluate_func_mapping = {
            "STATIC": self._static_evaluate,
            "DYNAMIC": self._dynamic_evaluate
        }
        evaluate_func = evaluate_func_mapping.get(self.env.ENV_CARD)
        if evaluate_func is None:
            raise ValueError("The env has unrecogonized ENV_CARD attribute")
        
        return evaluate_func()
    
    def _static_evaluate(self):
        
        accelerator = Accelerator()
        writer = create_tensorboard(save_dir=self.working_dir)
        
        batch_size = self.interaction_config.batch_size
        output_dir = self.interaction_config.output_dir

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()
        
        # construct eval recorder
        test_funcs = [self.env_cls.compute_reward]
        save_file = os.path.join(output_dir, "answer.json")
        recorder = StaticEvalRecorder(compute_metrics=test_funcs, writer=writer, log_file=save_file)
        
        # batch generation
        for test_batch in tqdm(test_dataloader):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                # construct InteractionDataProto object
                prompts = [x["prompt"] for x in test_batch]
                prompt_inputs = self.processing_class(
                    text=prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True
                )
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                gen_batch = InteractionDataProto()
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)
                gen_batch.no_tensor_batch["initial_prompts"] = [x["prompt"] for x in test_batch]

                # generation manager
                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)
            
                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            recorder.record_batch(completions, test_batch)
        recorder.finalize()
        writer.close()


    def _dynamic_evaluate(self):
        
        def _set_batch_envs(batch: list) -> tuple[list[str], list[str], list]:  # batch set envs
            system_prompts, init_user_prompts, envs = [], [], []
            for task_config in batch:
                env = self.env_cls(self.config.get("dataset"))
                system_prompt, init_user_prompt = env.set_env(task_config)

                system_prompts.append(system_prompt)
                init_user_prompts.append(init_user_prompt)
                envs.append(env)
            
            return system_prompts, init_user_prompts, envs
        
        def _build_data_proto(
            system_prompts: list[str], init_user_prompts: list[str], envs: list
        ) -> InteractionDataProto:
            messages = []
            for system_prmopt, init_user_prompt in zip(system_prompts, init_user_prompts):
                system_message = {"role": "system", "content": system_prmopt}
                user_message = {"role": "user", "content": init_user_prompt}
                init_messages = [system_message, user_message]
                messages.append(init_messages)

            data_proto = InteractionDataProto()
            data_proto.no_tensor_batch["init_prompts"] = messages
            data_proto.no_tensor_batch["envs"] = envs

            return data_proto
        
        # ===== body =====
        output_dir = self.interaction_config.output_dir

        accelerator = Accelerator()
        writer = create_tensorboard(save_dir=self.working_dir) 
        save_file = os.path.join(output_dir, "conversations.txt")
        recorder = DynamicEvalRecorder(writer=writer, log_file=save_file)

        batch_size = self.interaction_config.batch_size
        
        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()
        
        # batch generate
        for step, test_batch in tqdm(enumerate(test_dataloader)):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                system_prompts, init_user_prompts, envs = _set_batch_envs(test_batch) 
                input_data_proto = _build_data_proto(system_prompts, init_user_prompts, envs)
                
                self.generation_manager.actor_rollout_wg = unwrapped_model
                outputs: InteractionDataProto = self.generation_manager.run_agent_loop(input_data_proto)
                
                inter_histories = outputs.no_tensor_batch["inter_histories"]
                inter_context = self.processing_class.apply_chat_template(inter_histories, tokenize=False)

            # batch record
            rewards = []
            for env in input_data_proto.no_tensor_batch["envs"]:
                reward = env.feedback()
                rewards.append(reward)
            
            recorder.record_batch(inter_context, rewards)
        
        recorder.finalize()
        writer.close()
    
    def _parse_configs(self, configs):
        
        self.train_weaver = configs.get("train_weaver", True)
        self.train_trigger = configs.get("train_trigger", False)

        # --- parse weaver training args ---
        self.train_weaver_method = configs.get("train_weaver_method", "sft")
        if self.train_weaver_method not in ["sft", "grpo"]:
            raise ValueError("Unsupported weaver training method.")
        
        # parse weaver sft training args
        weaver_config = configs.get("weaver", dict())
        weaver_sft_config = weaver_config.get("sft", dict())
        self.weaver_sft_training_args = SFTConfig(**weaver_sft_config)
        self.weaver_sft_training_args.output_dir = os.path.join(self.working_dir, "weaver")

        # parse weaver grpo training args
        weaver_grpo_config = weaver_config.get("grpo", dict())
        self.weaver_grpo_training_args = GRPOConfig(**weaver_grpo_config)
        self.weaver_grpo_training_args.output_dir = os.path.join(self.working_dir, "weaver")

        # --- parse trigger training args ---
        trigger_config = configs.get("trigger", dict()) 
        self.train_trigger_method = configs.get("train_trigger_method", "grpo")
        if self.train_trigger_method not in ["grpo"]:
            raise ValueError("Unsupported trigger training method.")
        
        trigger_grpo_config = trigger_config.get("grpo", dict())
        self.trigger_grpo_training_args = GRPOConfig(**trigger_grpo_config)
        self.trigger_grpo_training_args.output_dir = os.path.join(self.working_dir, "trigger")

        # --- parse interaction args ---
        interaction_configs = configs.get("interaction", {})
        self.interaction_config = InteractionConfig(
            max_turns=interaction_configs.get("max_turns", 30),
            max_start_length=interaction_configs.get("max_start_length", 1024),
            max_prompt_length=interaction_configs.get("max_prompt_length", 4096),
            max_response_length=interaction_configs.get("max_response_length", 512),
            max_obs_length=interaction_configs.get("max_obs_length", 512),
            do_sample=interaction_configs.get("do_sample", False),
            temperature=interaction_configs.get("temperature", 1.0),
            batch_size=interaction_configs.get("batch_size", 32),
            output_dir=os.path.join(self.working_dir, "evaluate")
        )