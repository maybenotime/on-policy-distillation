#!/usr/bin/env python
"""Wrapper around `swift eval` for Vulcan cluster.

Vulcan compute nodes can reach huggingface.co but NOT modelscope.cn through
the Squid proxy whitelist. evalscope's TaskConfig defaults to ModelScope hub
and the dataset adapters hardcode `AI-ModelScope/<name>` IDs that don't
exist on HF. We patch both before invoking swift's eval entrypoint:

  1. TaskConfig.__init__ default `dataset_hub` → HuggingFace
  2. dataset_args injects HF-side dataset_id overrides for each benchmark

Use exactly like `swift eval ...`:
  python source/swift_eval_patched.py --use_hf true --model <ckpt> --eval_dataset gsm8k ...
"""

# Modelscope-style dataset_id → HuggingFace counterpart
HF_DATASET_ID_MAP = {
    "gsm8k":    "openai/gsm8k",
    "math_500": "HuggingFaceH4/MATH-500",
    "aime24":   "Maxwell-Jia/AIME_2024",
    "aime25":   "math-ai/aime25",
    "amc":      "AI-MO/aimo-validation-amc",
}


def _patch_evalscope():
    import evalscope.config
    from evalscope.constants import HubType

    _orig_init = evalscope.config.TaskConfig.__init__

    def _patched_init(self, *args, **kwargs):
        if "dataset_hub" not in kwargs:
            kwargs["dataset_hub"] = HubType.HUGGINGFACE
        # Inject HF dataset id overrides via dataset_args
        da = kwargs.get("dataset_args") or {}
        if isinstance(da, str):
            import json
            da = json.loads(da) if da else {}
        for dataset_name, hf_id in HF_DATASET_ID_MAP.items():
            entry = da.setdefault(dataset_name, {})
            entry.setdefault("dataset_id", hf_id)
        kwargs["dataset_args"] = da
        _orig_init(self, *args, **kwargs)

    evalscope.config.TaskConfig.__init__ = _patched_init


def main():
    _patch_evalscope()
    print("[patch] dataset_hub=huggingface, dataset_args=<HF id map>", flush=True)
    from swift.pipelines import eval_main
    eval_main()


if __name__ == "__main__":
    main()
