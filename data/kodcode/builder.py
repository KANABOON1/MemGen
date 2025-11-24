from datasets import DatasetDict, load_dataset
from typing import Dict
 
from data.base_builder import BaseBuilder
from data.kodcode.env import KodCodeEnv

class KodCodeBuilder(BaseBuilder):  
    
    def get_env_cls(self):
        return KodCodeEnv
    
    def _build_datasets(self) -> DatasetDict:
        # download dataset
        all_dataset = load_dataset("KodCode/KodCode-Light-RL-10K") 
        all_correct_dataset = all_dataset["train"]                 
        
        # train, valid, test dataset split
        train_ratio, valid_ratio, test_ratio = self.config.get("train_ratio"), self.config.get("valid_ratio"), self.config.get("test_ratio")
        assert train_ratio + valid_ratio + test_ratio == 1
        
        all_size = len(all_correct_dataset)
        test_size = int(all_size * test_ratio)
        split = all_correct_dataset.train_test_split(test_size=test_size, shuffle=True)
        train_valid_dataset, test_dataset = split["train"], split["test"]
        
        valid_size = int(len(train_valid_dataset) * valid_ratio / (train_ratio + valid_ratio))
        split = train_valid_dataset.train_test_split(test_size=valid_size, shuffle=True)
        train_dataset, valid_dataset = split["train"], split["test"]
        
        # preprocess
        train_dataset = train_dataset.map(self._sft_preprocess).select_columns(self._sft_keep_keys())
        valid_dataset = valid_dataset.map(self._sft_preprocess).select_columns(self._sft_keep_keys())
        test_dataset = test_dataset.map(self._sft_preprocess).select_columns(self._sft_keep_keys())
        
        # build dataset dict
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
    def _sft_preprocess(cls, example: Dict):
        
        format_template = r"Write an efficient and correct Python function to solve the following problem."
        prompt_template = "Question: {prompt}\n"

        question = example["question"].strip()
        solution = example["solution"].strip()

        processed_prompt = format_template + prompt_template.format(prompt=question)
        processed_label = solution

        text_output = {
            "prompt": processed_prompt,
            "completion": processed_label,
            "solution": processed_label,
            "test": example["test"].strip(),  
            "test_info": example["test_info"]
        }

        return text_output
    
    @classmethod
    def _sft_keep_keys(cls):
        return ["prompt", "completion", "solution", "test", "test_info"]


