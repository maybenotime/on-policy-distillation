"""5k answer-conditioned OPSD dataset plugin.

This plugin provides a runnable 5k online answer-conditioned distillation setup
for Chapter 5. It is a prompt-level late-intervention approximation: the teacher
copy receives the correct final answer and an intervention-ratio instruction.
Strict suffix-only KL masking still requires a trainer patch.
"""

import os
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

_RATIO = int(os.environ.get("LI_OPSD_RATIO", "70"))
assert 0 < _RATIO < 100, f"LI_OPSD_RATIO must be in (0, 100), got {_RATIO}"


def _student_messages(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def _build_teacher_prompt(problem: str, answer: str) -> str:
    remaining = 100 - _RATIO
    return (
        f"{problem}\n\n"
        f"The correct final answer is:\n{answer}\n\n"
        "You are acting as an answer-conditioned online distillation teacher. "
        f"Assume the student has already attempted roughly the first {_RATIO}% "
        "of the reasoning trajectory. Focus your supervision on the remaining "
        f"{remaining}%: correct possible drift, complete the derivation, and keep "
        "the reasoning concise and learnable for the student model. "
        "End with the final answer in \\boxed{}."
    )


class AnswerConditionedPreprocessor(RowPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        problem = row.get("problem", "")
        if not problem:
            problem = row.get("messages", [{}])[0].get("content", "")
        answer = row.get("reference_answer", row.get("answer", ""))

        problem = str(problem).strip()
        answer = str(answer).strip()
        if not problem or not answer:
            return None

        return {
            "messages": _student_messages(problem),
            "teacher_prompt": _build_teacher_prompt(problem, answer),
        }


_DATA_PATH = os.environ.get("LI_OPSD_DATA_PATH", "data/li_opsd_answer_first5k.jsonl")

register_dataset(
    DatasetMeta(
        dataset_name="local-li-opsd-math",
        dataset_path=_DATA_PATH,
        preprocess_func=AnswerConditionedPreprocessor(),
        tags=["math", "li-opsd", "answer-conditioned"],
    )
)
