from typing import Dict

def add_eos(example, eos_token):
    """在 labels 部分末尾添加 eos token
    """
    if "text" in example and not example["text"].endswith(eos_token):  
        example["text"] = example["text"] + eos_token
    elif "completion" in example and not example["completion"].endswith(eos_token):
        example["completion"] = example["completion"] + eos_token
    return example

def tokenize(example, processing_class) -> Dict:

    output = dict(example)
    prompt_ids = processing_class(
        text=example["prompt"], add_special_tokens=False
    )["input_ids"]
    completion_ids = processing_class(
        text=example["completion"], add_special_tokens=False
    )["input_ids"]
    input_ids = prompt_ids + completion_ids  
    
    # Create a completion mask
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    output["input_ids"] = input_ids
    output["completion_mask"] = completion_mask

    return output

def tokenize_instruction_example(example: Dict, processing_class) -> Dict:
    eos_token = processing_class.eos_token
    eos_example = add_eos(example, eos_token)
    tokenized_example = tokenize(eos_example, processing_class)
    
    return tokenized_example


def tokenize_conversation_example(example: Dict, processing_class) -> Dict:
    ...