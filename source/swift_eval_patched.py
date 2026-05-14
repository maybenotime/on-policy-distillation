#!/usr/bin/env python
"""Wrapper around `swift eval` for Vulcan cluster.

Vulcan compute nodes can reach huggingface.co but NOT modelscope.cn through
the Squid proxy whitelist. evalscope's TaskConfig defaults to ModelScope
(`evalscope/config.py:85`: `dataset_hub: str = HubType.MODELSCOPE`) and swift's
`--use_hf true` only sets `USE_HF=1` for model loading — it does NOT propagate
to evalscope's dataset hub. So without this wrapper, dataset loads hit
modelscope.cn and stall behind Squid.

We patch three things before invoking swift's eval entrypoint:

  1. TaskConfig.__init__ default `dataset_hub` → HuggingFace
  2. dataset_args injects HF-side dataset_id (and subset_list/eval_split)
     overrides for benchmarks whose ModelScope default has no HF counterpart
     with the right schema:
       - gsm8k     → openai/gsm8k                (lowercase question/answer)
       - math_500  → HuggingFaceH4/MATH-500      (lowercase problem/answer)
       - amc       → AI-MO/aimo-validation-amc + subset_list=['default'] +
                     eval_split='train' (evalscope's default
                     `evalscope/amc_22-24` does not exist on HF)
     aime24/aime25 are intentionally NOT overridden — their adapter defaults
     (`HuggingFaceH4/aime_2024`, `opencompass/AIME2025`) already work on HF
     with the right schema.
  3. AMCAdapter.record_to_sample is monkey-patched to use `.get()` for
     `year`/`url`. AI-MO/aimo-validation-amc only has [id, problem, answer,
     url] — no `year` column — so strict `record['year']` raises KeyError.
     We also disable split_as_subset since aimo-validation-amc has only one
     split (train).

Use exactly like `swift eval ...`:
  python source/swift_eval_patched.py --use_hf true --model <ckpt> --eval_dataset gsm8k aime24 aime25 amc math_500 ...
"""

HF_DATASET_ARGS = {
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
    },
    "math_500": {
        "dataset_id": "HuggingFaceH4/MATH-500",
    },
    "amc": {
        "dataset_id": "AI-MO/aimo-validation-amc",
        "subset_list": ["default"],
        "eval_split": "train",
    },
}


def _patch_amc_adapter():
    """AI-MO/aimo-validation-amc lacks `year`; adapter does strict
    `record['year']`. Switch to `.get()` and disable split_as_subset."""
    from evalscope.benchmarks.amc.amc_adapter import AMCAdapter
    from evalscope.api.dataset import Sample

    def _patched_record_to_sample(self, record):
        return Sample(
            input=record["problem"],
            target=record["answer"],
            metadata={
                "year": record.get("year", ""),
                "url": record.get("url", ""),
                "solution": record.get("solution", ""),
            },
        )

    AMCAdapter.record_to_sample = _patched_record_to_sample

    orig_init = AMCAdapter.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.split_as_subset = False

    AMCAdapter.__init__ = _patched_init


def _patch_evalscope():
    import evalscope.config
    from evalscope.constants import HubType

    _orig_init = evalscope.config.TaskConfig.__init__

    def _patched_init(self, *args, **kwargs):
        if "dataset_hub" not in kwargs:
            kwargs["dataset_hub"] = HubType.HUGGINGFACE
        da = kwargs.get("dataset_args") or {}
        if isinstance(da, str):
            import json
            da = json.loads(da) if da else {}
        for dataset_name, override in HF_DATASET_ARGS.items():
            entry = da.setdefault(dataset_name, {})
            for k, v in override.items():
                entry.setdefault(k, v)
        kwargs["dataset_args"] = da
        _orig_init(self, *args, **kwargs)

    evalscope.config.TaskConfig.__init__ = _patched_init


def main():
    _patch_evalscope()
    _patch_amc_adapter()
    print("[patch] dataset_hub=HuggingFace, dataset_args=<HF overrides>, AMCAdapter.record_to_sample uses .get()", flush=True)
    from swift.pipelines import eval_main
    eval_main()


if __name__ == "__main__":
    main()
