import torch
from transformers import PreTrainedTokenizerBase 
from typing import Dict, List, Any

class DynamicPaddingDataCollater:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):

        self.tokenizer = tokenizer

        if tokenizer.pad_token_id is None:
            print("Warning: Tokenizer does not have a pad_token_id. Using 0 for input_ids and attention_mask padding.")
            self.padding_value_input = 0
        else:
            self.padding_value_input = tokenizer.pad_token_id

        # labels 的填充值
        self.padding_value_label = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        processed_features = []
        for feature in features:
            input_ids = feature["input_ids"]
            completion_mask = feature["completion_mask"]

            prompt_ids = [token for token, is_completion in zip(input_ids, completion_mask) if not is_completion]

            label_ids = [token for token, is_completion in zip(input_ids, completion_mask) if is_completion]

            processed_features.append({
                "prompt_ids": prompt_ids,
                "label_ids": label_ids,

                "original": feature 
            })

        max_prompt_len = max(len(f["prompt_ids"]) for f in processed_features)
        max_label_len = max(len(f["label_ids"]) for f in processed_features)

        padded_prompt_ids = []
        padded_input_attention_mask = []
        padded_label_ids = []
        padded_labels_attention_mask = []

        for feature in processed_features:

            prompt_ids = feature["prompt_ids"]
            label_ids = feature["label_ids"]
            

            num_input_pads = max_prompt_len - len(prompt_ids)
            padded_prompt_ids.append([self.padding_value_input] * num_input_pads + prompt_ids)

            input_attention_mask = [1] * len(prompt_ids)
            num_input_mask_pads = max_prompt_len - len(input_attention_mask)
            padded_input_attention_mask.append([0] * num_input_mask_pads + input_attention_mask)

            num_label_pads = max_label_len - len(label_ids)
            padded_label_ids.append(label_ids + [self.padding_value_label] * num_label_pads)
            
            labels_attention_mask = [1] * len(label_ids)
            num_label_mask_pads = max_label_len - len(labels_attention_mask)
            padded_labels_attention_mask.append(labels_attention_mask + [0] * num_label_mask_pads)
        
        batch = {
            "prompt_ids": torch.tensor(padded_prompt_ids, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(padded_input_attention_mask, dtype=torch.long),
            "label_ids": torch.tensor(padded_label_ids, dtype=torch.long),
            "label_attention_mask": torch.tensor(padded_labels_attention_mask, dtype=torch.long),
        }
        
        batch["raw_samples"] = [f["original"] for f in processed_features]

        return batch

        


