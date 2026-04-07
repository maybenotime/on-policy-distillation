"""OPSD dataset plugin for math_all_teachers_correct.jsonl.

On-Policy Self-Distillation: student sees only the problem; teacher sees
problem + a correct reference solution as privileged information.

Three selection modes, controlled by the OPSD_MODE environment variable:

  single (default)
      Use the answer from ONE specific teacher model.
      Set the model with OPSD_TEACHER_MODEL (default: deepseek-r1).
      Records where that model answered incorrectly are skipped.

  all_correct
      Concatenate every correct teacher model's answer into one privileged
      context.  Records where NO model answered correctly are skipped.

  multi
      Concatenate correct answers from a user-specified subset of models.
      Set the subset with OPSD_TEACHER_MODELS (comma-separated, e.g.
      "deepseek-r1,Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct").
      Records where none of the specified models answered correctly are skipped.

Usage:
    OPSD_MODE=single OPSD_TEACHER_MODEL=Qwen2.5-72B-Instruct \\
        swift rlhf --rlhf_type gkd --external_plugins source/opsd_plugin.py ...

    OPSD_MODE=all_correct \\
        swift rlhf --rlhf_type gkd --external_plugins source/opsd_plugin.py ...

    OPSD_MODE=multi OPSD_TEACHER_MODELS=deepseek-r1,Qwen2.5-7B-Instruct,Llama-3.1-8B-Instruct \\
        swift rlhf --rlhf_type gkd --external_plugins source/opsd_plugin.py ...
"""

import os
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = 'Please reason step by step, and put your final answer within \\boxed{}.'

TRANSITION_PROMPT = (
    'After understanding the reference solution and the rationale behind each step, '
    'now articulate your own step-by-step reasoning that derives the final answer.'
)

# ── All teacher model keys present in the dataset ─────────────────────────────

ALL_TEACHER_MODELS: List[str] = [
    'gemma2-9b-it',
    'gemma2-27b-it',
    'Llama-3.1-8B-Instruct',
    'Llama-3.3-70B-Instruct',
    'Mistral-7B-Instruct',
    'Mistral-Nemo-Instruct-2407',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-14B-Instruct',
    'Qwen2.5-32B-Instruct',
    'Qwen2.5-72B-Instruct',
    'Qwen2.5-Math-7B-Instruct',
    'Qwen3-8B',
    'Qwen3-14B',
    'deepseek-r1-distill-qwen-7B',
    'deepseek-r1',
]

# ── Config from environment ────────────────────────────────────────────────────

_MODE = os.environ.get('OPSD_MODE', 'single').strip().lower()
_TEACHER_MODEL = os.environ.get('OPSD_TEACHER_MODEL', 'deepseek-r1').strip()
_TEACHER_MODELS: List[str] = [
    m.strip() for m in os.environ.get('OPSD_TEACHER_MODELS', '').split(',') if m.strip()
]

assert _MODE in ('single', 'all_correct', 'multi'), (
    f"OPSD_MODE must be 'single', 'all_correct', or 'multi', got '{_MODE}'"
)
if _MODE == 'multi':
    assert _TEACHER_MODELS, (
        "OPSD_MODE=multi requires OPSD_TEACHER_MODELS to be set "
        "(comma-separated list of model keys, e.g. 'deepseek-r1,Qwen2.5-7B-Instruct')"
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_teacher_prompt(problem: str, solution_block: str) -> str:
    return (
        f'{problem}\n\n'
        f'Here is a reference solution to this problem:\n{solution_block}\n\n'
        f'{TRANSITION_PROMPT}'
    )


def _student_messages(problem: str) -> List[Dict[str, str]]:
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': problem},
    ]


# ── Preprocessors ──────────────────────────────────────────────────────────────

class SingleTeacherPreprocessor(RowPreprocessor):
    """Use one specific teacher model's correct answer as privileged info.

    Interface:
        OPSD_TEACHER_MODEL  — model key in the dataset (default: deepseek-r1)

    Records where the selected model's answer is incorrect are skipped (return None).
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        model_data = row.get(_TEACHER_MODEL)
        if not model_data:
            return None  # key not present in this record
        if not model_data.get('correctness_math_verify', False):
            return None  # model answered incorrectly — skip

        problem = row.get('messages', [{}])[0].get('content', row.get('problem', ''))
        solution = model_data['generation']

        return {
            'messages': _student_messages(problem),
            'teacher_prompt': _build_teacher_prompt(problem, solution),
        }


class AllCorrectTeachersPreprocessor(RowPreprocessor):
    """Concatenate every correct teacher model's answer as privileged info.

    Interface:
        (no extra env var needed — uses all models in ALL_TEACHER_MODELS)

    Records where NO model answered correctly are skipped (return None).
    The combined teacher_prompt lists each model's answer under a labelled header.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        problem = row.get('messages', [{}])[0].get('content', row.get('problem', ''))

        correct_answers: List[str] = []
        for model_key in ALL_TEACHER_MODELS:
            model_data = row.get(model_key)
            if model_data and model_data.get('correctness_math_verify', False):
                correct_answers.append(
                    model_data['generation']
                )

        if not correct_answers:
            return None  # no model got it right — skip

        solution_block = '\n\n'.join(correct_answers)

        return {
            'messages': _student_messages(problem),
            'teacher_prompt': _build_teacher_prompt(problem, solution_block),
        }


class MultiTeacherPreprocessor(RowPreprocessor):
    """Concatenate correct answers from a user-specified subset of teacher models.

    Interface:
        OPSD_TEACHER_MODELS  — comma-separated model keys (required)

    Records where none of the specified models answered correctly are skipped.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        problem = row.get('messages', [{}])[0].get('content', row.get('problem', ''))

        correct_answers: List[str] = []
        for model_key in _TEACHER_MODELS:
            model_data = row.get(model_key)
            if model_data and model_data.get('correctness_math_verify', False):
                correct_answers.append(model_data['generation'])

        if not correct_answers:
            return None

        solution_block = '\n\n'.join(correct_answers)

        return {
            'messages': _student_messages(problem),
            'teacher_prompt': _build_teacher_prompt(problem, solution_block),
        }


# ── Dataset registration ───────────────────────────────────────────────────────

if _MODE == 'single':
    _preprocessor = SingleTeacherPreprocessor()
elif _MODE == 'multi':
    _preprocessor = MultiTeacherPreprocessor()
else:
    _preprocessor = AllCorrectTeachersPreprocessor()

_DATA_PATH = os.environ.get(
    'OPSD_DATA_PATH',
    '/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl',
)

register_dataset(
    DatasetMeta(
        dataset_name='local-opsd-math',
        dataset_path=_DATA_PATH,
        preprocess_func=_preprocessor,
        tags=['math', 'opsd'],
    )
)
