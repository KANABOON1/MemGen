from typing import List, Dict, Tuple
import re

from data.base_env import DynamicEnv
from data.utils.retrieval_utils import Retriever

class TriviaQAEnv(DynamicEnv):
   
    def __init__(self, configs: Dict):
        super().__init__(configs)
        self.explorer = Retriever()

    def set_env(self, task_config: Dict) -> None:
        if task_config.get('answer') is None:
            raise ValueError('Please provide the answer for the task')
        if task_config.get("prompt") is None:
            raise ValueError('Please provide the prompt for the task')

        self.task_config = task_config
        
        self._reset()

        from data.triviaqa.builder import TRIVIAQA_SYSTEM_PROMPT
        return TRIVIAQA_SYSTEM_PROMPT, task_config["prompt"]
    
    def _reset(self):
        self.done = False
        self.reward = 0.0

    def step(self, action: str) -> Tuple[str, float, bool]:
        action = self.preprocess_action(action)
        action_type, action_content = self._process_action(action)
        observation = None
        
        if action_type == "search":
            try:
                observation = self.explorer.batch_search([action_content])[0]
            except Exception as e:
                observation = f'Cannot find corresponding pages.'     
            self.done = False
            self.reward = 0.0

        elif action_type == "answer":
            observation = ""
            self.done = True
            self.reward = 1.0 if self._check_answer(action_content, self.task_config["answer"]) else 0.0
        else:
            observation = "\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
            self.done = False
            self.reward = 0.0

        return observation, self.reward, self.done
    
    @classmethod
    def preprocess_action(cls, action: str) -> str:
        if "</search>" in action:
            return action.split("</search>", 1)[0] + "</search>"
        elif "</answer>" in action:
            return action.split("</answer>", 1)[0] + "</answer>"
        else:
            return action
    
    @classmethod
    def _process_action(cls, action: str):
        action = action.strip()

        if "<search>" in action and "</search>" in action:
            start = action.index("<search>") + len("<search>")
            end = action.index("</search>")
            content = action[start:end].strip().split("\n", 1)[0].strip()
            return "search", content

        if "<answer>" in action and "</answer>" in action:
            start = action.index("<answer>") + len("<answer>")
            end = action.index("</answer>")
            content = action[start:end].strip().split("\n", 1)[0].strip()
            return "answer", content

        return "think", action
    
    def _check_answer(self, answer: str, ground_truth: List[str]):
        answer = answer.lower()
        for gt in ground_truth:
            if gt.lower() in answer:
                return True
        
        return False
    
    def feedback(self) -> float:
        return self.reward

    @classmethod
    def compute_reward(cls, completions: List[str], envs: List['TriviaQAEnv'], **kwargs) -> List[float]:
        scores = []
        for completion, env in zip(completions, envs):
            solution = env.task_config['answer']  

            matches = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            
            if not matches:  
                scores.append(0.0)
                continue

            extracted = matches[-1].strip()

            correct = False
            for s in solution:
                if s.lower() in extracted.lower():
                    correct = True
                    break

            if correct:
                scores.append(1.0)
            else:
                scores.append(0.5)

        return scores

