import random
from datasets import DatasetDict, load_dataset
 
from data.base_builder import BaseBuilder
from data.gpqa.env import GPQAEnv

class GPQABuilder(BaseBuilder):

    def get_env_cls(self):
        return GPQAEnv
    
    def _build_datasets(self) -> DatasetDict:
        # download data
        raw_train_dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
        raw_test_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        val_size = int(len(raw_train_dataset) * self.config.get("valid_ratio"))
        split = raw_train_dataset.train_test_split(test_size=val_size, shuffle=True)
        raw_train_dataset, raw_valid_dataset = split["train"], split["test"]
        
        # preprocess
        train_dataset = raw_train_dataset.map(self._preprocess).select_columns(self._keep_keys())
        valid_dataset = raw_valid_dataset.map(self._preprocess).select_columns(self._keep_keys())
        test_dataset = raw_test_dataset.map(self._preprocess).select_columns(self._keep_keys())

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
    def _preprocess(cls, example: dict):
        
        def build_answer_map(candidates: list[str]) -> dict[str, dict[str, object]]:

            indices = list(range(len(candidates)))
            random.shuffle(indices)

            orders = [chr(ord("A") + i) for i in range(len(candidates))]

            answer_map = {}
            for idx, candidate_idx in enumerate(indices):
                answer = candidates[candidate_idx]
                answer_map[answer] = {
                    "order": orders[idx],
                    "is_correct": (candidate_idx == 0) 
                }

            return answer_map
        
        def build_question(question, answer_map: dict) -> str:
            result = question.strip() + "\n\nPlease choose one of the following options:\n"

            sorted_items = sorted(answer_map.items(), key=lambda x: x[1]["order"])

            for answer, meta in sorted_items:
                result += f"{meta['order']}. {answer}\n"

            return result

        def build_answer(rationale: str, answer_map: dict) -> str:
            correct_answer = None
            for key, value in answer_map.items():
                if value.get("is_correct") is True:
                    correct_answer = value.get("order")
            assert correct_answer is not None
            return rationale + f"\n\nTherefore, the final answer is \\boxed{{{correct_answer}}}"

        question = example["Question"].strip()
        explanation = example["Explanation"].strip()
        correct_answer = example["Correct Answer"].strip()
        incorrect_answer1 = example["Incorrect Answer 1"].strip() 
        incorrect_answer2 = example["Incorrect Answer 2"].strip()
        incorrect_answer3 = example["Incorrect Answer 3"].strip()
        
        answers_map = build_answer_map([correct_answer, incorrect_answer1, incorrect_answer2, incorrect_answer3])
        question = build_question(question, answers_map)
        answer = build_answer(explanation, answers_map)

        format_template = r"""Solve the problem with proper reasoning, and make sure to put the FINAL CHOICE inside \boxed{}."""
        prompt_template = "Question: {prompt}\n"   
        processed_prompt = format_template + prompt_template.format(prompt=question)

        text_output = {
            "prompt": processed_prompt,
            "completion": answer,
            "solution": answer    
        }
        return text_output
    
    @classmethod
    def _keep_keys(cls):
        return ["prompt", "completion", "solution"]