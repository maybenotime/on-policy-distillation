"""evalscope DataAdapter for `MathArena/hmmt_feb_2026` (HMMT February 2026, 33 problems).

Schema (HF dataset):
  problem_idx: int
  problem: str            (LaTeX problem statement)
  answer: str             (LaTeX final answer)
  problem_type: list[str]

Mirrors `aime25_adapter.py` — same boxed-answer + math_verify-style grading.
Importing this module triggers `register_benchmark` so `--eval_dataset hmmt_feb_2026`
works after `import source.hmmt_adapter`.
"""
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.api.evaluator.state import TaskState

PROMPT_TEMPLATE = """{question}
Please reason step by step, and put your final answer within \\boxed{{}}."""


@register_benchmark(
    BenchmarkMeta(
        name='hmmt_feb_2026',
        pretty_name='HMMT-February-2026',
        description=(
            'Harvard-MIT Mathematics Tournament, February 2026 (33 problems). '
            'Latex problem + latex boxed answer. Graded with math_verify-style '
            'answer matching, mirroring AIME25 adapter.'
        ),
        dataset_id='MathArena/hmmt_feb_2026',
        subset_list=['default'],
        metric_list=[{'acc': {'numeric': True}}],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # the only split — 33 rows
        prompt_template=PROMPT_TEMPLATE,
    )
)
class HMMTFeb2026Adapter(DefaultDataAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(input=record['problem'], target=record['answer'])

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        from evalscope.metrics.math_parser import extract_answer as _extract
        from evalscope.benchmarks.aime.math_normalize import normalize_answer

        return normalize_answer(_extract(prediction))

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        from evalscope.benchmarks.aime.grader import grade_answer
        from evalscope.metrics.math_parser import extract_answer as _extract

        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        try:
            score.value['acc'] = 1.0 if grade_answer(_extract(original_prediction), reference) else 0.0
        except Exception as e:
            score.value['acc'] = 0.0
            score.metadata['acc'] = f'grading_error: {e}'
        return score
