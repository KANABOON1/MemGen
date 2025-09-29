from typing import Dict, List
import re

from larm.data.envs.base_env import StaticEnv
from larm.common.registry import registry
from larm.data.utils.code_utils import PyExecutor, extract_python_code



@registry.register_env("kodcode")
class KodCodeEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def _rename_func(cls, answer: str, function_name: str) -> str:
        """
        Replace the name of the first function in `answer` with `function_name`.
        Only modifies the function name, keeps everything else intact.
        """
        pattern = r"def\s+(\w+)\s*\("

        new_answer = re.sub(pattern, f"def {function_name}(", answer, count=1)
        return new_answer

    @classmethod
    def _accuracy_reward(cls, completions: List[str], test: List[str], test_info, **kwargs) -> List[float]:
 
        py_executor = PyExecutor()
        scores = []
        for completion, t, tf in zip(completions, test, test_info): 
            func_blocks = extract_python_code(completion.strip())
            collected_answer = '\n'.join(func_blocks)
            renamed_answer = cls._rename_func(collected_answer, tf[0]["function_name"])
            _, _, results = py_executor.execute(renamed_answer, [t])

            score = sum(results) / len(results)  
            scores.append(score)
        
        return scores

    @classmethod
    def _format_reward(cls, completions: List[str], **kwargs):
        pass
