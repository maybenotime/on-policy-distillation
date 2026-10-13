"""Constant-zero reward for pure on-policy distillation.

verl's default_compute_score dispatches on `data_source` and raises
NotImplementedError for anything it doesn't know. Our OpenThoughts3 code prompts
carry no test cases, so no rule-based score is possible — and none is needed:
distillation runs with use_task_rewards=False / use_policy_gradient=False, so
the KL against the teacher is the entire objective and the reward channel is
inert. Returning a constant keeps the agent-loop reward path alive without
pretending to grade anything.

Wire it up with:
  reward.custom_reward_function.path=/home/sya225/opd/scripts/zero_reward.py
  reward.custom_reward_function.name=compute_score
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return 0.0
