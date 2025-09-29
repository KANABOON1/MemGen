import os
from datasets import DatasetDict, load_dataset, Dataset
from typing import Dict
import random
 
from larm.data.builders.base_builder import BaseDatasetBuilder
from larm.data.interactions.singleturn_interaction import SingleTurnInteractionManager
from larm.common.registry import registry
from larm.data.envs.gsm8k_env import GSM8KEnv

@registry.register_builder("gsm8k")
class GSM8KBuilder(BaseDatasetBuilder):   # Env

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gsm8k/default.yaml",
    }
    CACHE_PATH = None
    
    def _build_datasets(self) -> DatasetDict:

        # download data
        raw_dataset = load_dataset("gsm8k", "main")
        raw_train_dataset, raw_test_dataset = raw_dataset['train'], raw_dataset['test']
        val_size = int(len(raw_train_dataset) * self.config.val_ratio)
        split = raw_train_dataset.train_test_split(test_size=val_size, shuffle=True)
        raw_train_dataset, raw_valid_dataset = split["train"], split["test"]
        
        # save to json
        cache_root = registry.get_path("cache_root")
        save_dir = os.path.join(cache_root, "gsm8k")
        raw_train_dataset.to_json(os.path.join(save_dir, "train.json"))
        raw_valid_dataset.to_json(os.path.join(save_dir, "valid.json"))
        raw_test_dataset.to_json(os.path.join(save_dir, "test.json"))
        
        # preprocess
        num_workers = 32
        train_dataset = raw_train_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        valid_dataset = raw_valid_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        test_dataset = raw_test_dataset.map(self._preprocess, num_proc=num_workers).select_columns(self._keep_keys())
        
        # build dataset
        dataset_dict = DatasetDict()
        dataset_dict["train"] = train_dataset
        dataset_dict["valid"] = valid_dataset
        dataset_dict["test"] = test_dataset

        return dataset_dict
    
    def _build_sft_datasets(self) -> DatasetDict:
        return self._build_datasets()


    def _build_rl_datasets(self) -> DatasetDict:
        return self._build_datasets()
    
    @classmethod
    def _preprocess(cls, example: Dict):
        def _preprocess_answer(answer: str) -> str:
            raw_answer_list = answer.split("\n####")
            rationale = raw_answer_list[0]
            clean_answer = raw_answer_list[-1].strip()
            boxed_answer = "\\boxed{" + clean_answer + "}"
            new_string = rationale + boxed_answer
            return new_string.strip()
        
        format_template = r"""Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."""
        prompt_template = "Question: {prompt}\n"
         
        question = example["question"].strip()
        answer = example["answer"].strip()

        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = _preprocess_answer(answer)

        text_output = {
            "prompt": processed_prompt,
            "completion": processed_label,
            "solution": processed_label,     
            "test": processed_label,
        }
        
        # NOTE - To use the built-in tokenization mechanism of SFTTrainer, 
        # it is necessary to ensure that the prompt + completion is lossless.
        return text_output
    
    @classmethod
    def _keep_keys(cls):
        return ["prompt", "completion", "solution"]
    
    def get_env_cls(self):
        return GSM8KEnv
        
    def get_generation_manager_cls(self):
        return SingleTurnInteractionManager