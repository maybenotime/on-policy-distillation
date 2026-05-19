#!/usr/bin/env python
"""Wrapper around `swift eval` for Vulcan cluster.

Vulcan compute nodes can reach huggingface.co but NOT modelscope.cn through
the Squid proxy whitelist. evalscope's TaskConfig defaults to ModelScope
(`evalscope/config.py:85`: `dataset_hub: str = HubType.MODELSCOPE`) and swift's
`--use_hf true` only sets `USE_HF=1` for model loading — it does NOT propagate
to evalscope's dataset hub.

We patch four things before invoking swift's eval entrypoint:

  1. TaskConfig.__init__ default `dataset_hub` → HuggingFace
  2. dataset_args injects HF-side dataset_id (and subset_list/eval_split)
     overrides for gsm8k/math_500. aime24/aime25 use adapter defaults
     (already HF). amc DOES use evalscope's own `evalscope/amc_22-24`
     which is modelscope-only — see point 4.
  3. AMCAdapter.record_to_sample uses `.get()` for `year`/`url` so it
     tolerates whichever amc dataset variant we serve.
  4. modelscope `MsDataset.load` is monkey-patched to short-circuit on
     `evalscope/amc_22-24` and read directly from the local arrow cache
     pre-downloaded on the login node. This lets us use the canonical
     evalscope amc benchmark (amc22+amc23+amc24, 163 problems) without
     hitting modelscope.cn from compute nodes.

Use exactly like `swift eval ...`:
  python source/swift_eval_patched.py --use_hf true --model <ckpt> --eval_dataset gsm8k aime24 aime25 amc math_500 ...
"""

import glob
import os

# evalscope's amc adapter uses the modelscope-only `evalscope/amc_22-24`
# dataset by default — we keep it that way (no dataset_id override here)
# and serve it from the pre-downloaded local arrow cache via the
# MsDataset.load monkey-patch below.
HF_DATASET_ARGS = {
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
    },
    "math_500": {
        "dataset_id": "HuggingFaceH4/MATH-500",
    },
}

# Pre-downloaded arrow cache locations (populated on Vulcan login node
# via modelscope SDK — login node CAN reach modelscope.cn).
_AMC_LOCAL_CACHE = (
    "/scratch/sya225/hf_cache/modelscope/datasets/evalscope___amc_22-24/"
    "default-c03498f62d9e7246/0.0.0/master"
)
_LOCAL_DATASETS = {
    "evalscope/amc_22-24": _AMC_LOCAL_CACHE,
}


def _patch_amc_adapter():
    """`evalscope/amc_22-24` records have `year`/`url`, but be defensive
    in case a different amc source ever gets wired in."""
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


def _patch_msdataset_load():
    """Make `MsDataset.load(dataset_name='evalscope/amc_22-24', ...)` read
    from the local arrow cache instead of hitting modelscope.cn.

    The cached `master/` dir contains one arrow file per split:
      amc_22-24-amc22.arrow, amc_22-24-amc23.arrow, amc_22-24-amc24.arrow,
      amc_22-24-test.arrow
    """
    import modelscope.msdatasets.ms_dataset as ms_module
    import datasets as hf_datasets

    _orig_load = ms_module.MsDataset.load

    def _patched_load(dataset_name=None, **kwargs):
        if dataset_name in _LOCAL_DATASETS:
            local_dir = _LOCAL_DATASETS[dataset_name]
            split = kwargs.get("split") or kwargs.get("subset_name") or "test"
            # arrow file pattern: <dataset_basename>-<split>.arrow
            basename = dataset_name.split("/")[-1]  # 'amc_22-24'
            arrow_file = os.path.join(local_dir, f"{basename}-{split}.arrow")
            if not os.path.exists(arrow_file):
                # fall back to scanning the dir
                candidates = glob.glob(os.path.join(local_dir, f"*-{split}.arrow"))
                if candidates:
                    arrow_file = candidates[0]
            if not os.path.exists(arrow_file):
                # Fall back to the real modelscope download (works on hosts
                # like fir that can reach modelscope.cn).
                print(
                    f"[swift_eval_patched] local arrow file not found for "
                    f"{dataset_name}/{split} (tried {arrow_file}); falling "
                    f"back to MsDataset.load over the network.",
                    flush=True,
                )
                return _orig_load(dataset_name=dataset_name, **kwargs)
            print(
                f"[swift_eval_patched] MsDataset.load → local arrow: "
                f"{arrow_file}",
                flush=True,
            )
            return hf_datasets.Dataset.from_file(arrow_file)
        return _orig_load(dataset_name=dataset_name, **kwargs)

    ms_module.MsDataset.load = _patched_load


def _patch_evalscope():
    import evalscope.config
    from evalscope.constants import HubType

    _orig_init = evalscope.config.TaskConfig.__init__
    pass_at_k = int(os.environ.get("PASS_AT_K", "0") or "0")

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
        amc_entry = da.setdefault("amc", {})
        amc_entry.setdefault("dataset_id", "evalscope/amc_22-24")
        # pass@k mode: override aggregation per-dataset + set repeats/gen_config
        if pass_at_k > 1:
            for ds in ("gsm8k", "aime24", "aime25", "amc", "math_500"):
                da.setdefault(ds, {})["aggregation"] = "mean_and_pass_at_k"
            kwargs["repeats"] = pass_at_k
            gc = kwargs.get("generation_config") or {}
            if isinstance(gc, str):
                import json
                gc = json.loads(gc) if gc else {}
            gc.setdefault("temperature", 0.7)
            gc.setdefault("top_p", 0.95)
            gc.setdefault("max_tokens", 4096)
            kwargs["generation_config"] = gc
        kwargs["dataset_args"] = da
        _orig_init(self, *args, **kwargs)

    evalscope.config.TaskConfig.__init__ = _patched_init


def _patch_eval_loader():
    """evalscope's RemoteDataLoader uses one global data_source (from
    TaskConfig.dataset_hub) for ALL benchmarks. We want HuggingFace for
    gsm8k/math_500/aime24/aime25 but MODELSCOPE for amc (so the local
    arrow short-circuit in MsDataset.load kicks in). Override the load()
    entry point to flip self.data_source to MODELSCOPE for amc only,
    then delegate to the original."""
    import evalscope.api.dataset.loader as loader_mod
    from evalscope.constants import HubType

    orig_load = loader_mod.RemoteDataLoader.load

    def _patched_load(self):
        if self.data_id_or_path == "evalscope/amc_22-24":
            self.data_source = HubType.MODELSCOPE
        return orig_load(self)

    loader_mod.RemoteDataLoader.load = _patched_load


def main():
    _patch_evalscope()
    _patch_amc_adapter()
    _patch_msdataset_load()
    _patch_eval_loader()
    # Import custom adapters so their @register_benchmark fires.
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent))
    try:
        import hmmt_adapter  # noqa: F401
    except ImportError as _e:
        # hmmt_adapter.py is untracked on some clusters; skip silently so
        # eval runs even without HMMT support.
        print(f"[patch] hmmt_adapter not available: {_e}", flush=True)
    print(
        "[patch] dataset_hub=HF (amc forced to MODELSCOPE in loader.load), "
        "dataset_args=<HF overrides>, "
        "MsDataset.load short-circuited for evalscope/amc_22-24, "
        "AMCAdapter.record_to_sample uses .get()",
        flush=True,
    )
    from swift.pipelines import eval_main
    eval_main()


if __name__ == "__main__":
    main()
