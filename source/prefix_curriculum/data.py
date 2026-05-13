"""Dataset plugin for TP-BPOD (backward-direction prefix curriculum).

Loads `final_math_dataset_updated.jsonl`, selects one teacher's pre-generated
trajectory per row, applies correctness + completeness filters, tokenizes the
trajectory with the *student* tokenizer, and registers the dataset under the
name `prefix-curriculum-backward-math` so `swift rlhf ... --dataset
prefix-curriculum-backward-math` finds it.

Environment variables:
    PREFIX_TEACHER          Teacher column to use (default: Qwen2.5-14B-Instruct)
    PREFIX_FILTER_CORRECT   1 = drop rows where teacher got the answer wrong (default 1)
    PREFIX_FILTER_COMPLETE  1 = drop rows where teacher rollout was truncated (default 1)
    PREFIX_DATA_PATH        Override path to the jsonl (default: see DEFAULT_DATA_PATH)
    PREFIX_STUDENT_MODEL    Student model id, used to load the tokenizer for
                            pre-tokenizing teacher trajectories. Required.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = (
    '/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl'
)

SYSTEM_PROMPT = (
    'Please reason step by step, and put your final answer within \\boxed{}.'
)

_TEACHER = os.environ.get('PREFIX_TEACHER', 'Qwen2.5-14B-Instruct').strip()
_FILTER_CORRECT = os.environ.get('PREFIX_FILTER_CORRECT', '1').strip() == '1'
_FILTER_COMPLETE = os.environ.get('PREFIX_FILTER_COMPLETE', '1').strip() == '1'
_DATA_PATH = os.environ.get('PREFIX_DATA_PATH', DEFAULT_DATA_PATH).strip()
_STUDENT_MODEL = os.environ.get('PREFIX_STUDENT_MODEL', '').strip()

if not _STUDENT_MODEL:
    raise RuntimeError(
        'PREFIX_STUDENT_MODEL must be set to the student model id so the data '
        'plugin can load the tokenizer that will be used to pre-tokenize the '
        'teacher trajectories.'
    )

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        logger.info('[prefix_curriculum.data] loading tokenizer: %s', _STUDENT_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(_STUDENT_MODEL, use_fast=True)
    return _tokenizer


class BackwardCurriculumPreprocessor(RowPreprocessor):
    """Emit one training row per qualifying record.

    Output schema:
        messages: [{system}, {user with problem}]
        teacher_full_token_ids: list[int] — full teacher rollout tokens
        answer: str — ground-truth answer (for downstream eval)
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        teacher_field = row.get(_TEACHER)
        if not teacher_field or not isinstance(teacher_field, dict):
            return None

        if _FILTER_CORRECT and not teacher_field.get('correctness_math_verify', False):
            return None
        if _FILTER_COMPLETE and not teacher_field.get('is_reasoning_complete', False):
            return None

        generation = teacher_field.get('generation')
        if not generation or not isinstance(generation, str):
            return None

        problem = row.get('problem') or row.get('prompt')
        if not problem:
            return None

        tokenizer = _get_tokenizer()
        teacher_token_ids = tokenizer.encode(generation, add_special_tokens=False)
        if not teacher_token_ids:
            return None

        return {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': problem},
            ],
            'teacher_full_token_ids': teacher_token_ids,
            'answer': str(row.get('answer', '')),
        }


def register():
    logger.info(
        '[prefix_curriculum.data] registering prefix-curriculum-backward-math'
        ' (teacher=%s, filter_correct=%s, filter_complete=%s, path=%s)',
        _TEACHER, _FILTER_CORRECT, _FILTER_COMPLETE, _DATA_PATH,
    )
    register_dataset(
        DatasetMeta(
            dataset_name='prefix-curriculum-backward-math',
            dataset_path=_DATA_PATH,
            preprocess_func=BackwardCurriculumPreprocessor(),
            tags=['math', 'prefix-curriculum', 'backward'],
        )
    )
