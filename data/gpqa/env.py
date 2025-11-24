from data.utils.math_utils import compute_score
from data.base_env import StaticEnv

class GPQAEnv(StaticEnv):

    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def compute_reward(cls, completions: list[str], solution: list[str], **kwargs) -> list[float]:
        
        scores = [compute_score(completion=c, ground_truth=s) for c, s in zip(completions, solution)]
        return scores

