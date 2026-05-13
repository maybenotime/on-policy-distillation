"""vLLM /v1/completions client for TP-BPOD rollout.

The vLLM rollout server (started by `scripts/rollout.sh` and used by Phase 1
SP-FPOD via `/v1/chat/completions`) also exposes `/v1/completions`, which
accepts raw token IDs as the `prompt` field. For TP-BPOD we need to send
`prompt = chat_template_prompt_ids + teacher_prefix_ids` so the student
generates conditioned on the teacher's partial solution.

This module adapts the pattern from `swift/rlhf_trainers/gkd_trainer.py:714`
(`fetch_teacher_logprobs`) for generation rather than logprob extraction.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)

_MODEL_NAME_CACHE: Optional[str] = None


def _resolve_model_name(base_url: str, timeout: float = 10.0) -> str:
    """Discover the model id served by the vLLM rollout server."""
    global _MODEL_NAME_CACHE
    if _MODEL_NAME_CACHE is not None:
        return _MODEL_NAME_CACHE
    try:
        resp = requests.get(f'{base_url}/v1/models', timeout=timeout)
        _MODEL_NAME_CACHE = resp.json()['data'][0]['id'] if resp.ok else 'default'
    except Exception:
        _MODEL_NAME_CACHE = 'default'
    return _MODEL_NAME_CACHE


def vllm_complete(
    base_url: str,
    prompt_token_ids_list: List[List[int]],
    max_tokens: Union[int, Sequence[int]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    timeout: float = 600.0,
    max_workers: int = 8,
) -> List[List[int]]:
    """POST raw token_ids to vLLM /v1/completions; return generated token_ids.

    Args:
        base_url: vLLM server base URL, e.g. 'http://127.0.0.1:8000'.
        prompt_token_ids_list: List of per-sample prompt token-id sequences.
            For TP-BPOD each is `chat_template_prompt_ids + teacher_prefix_ids`.
        max_tokens: Per-sample generation budget. Either a single int applied
            to every sample, or a sequence of length `len(prompt_token_ids_list)`
            specifying a per-sample cap. For TP-BPOD the caller should pass
            per-sample caps equal to `len(teacher_full_tokens) - Q_i`.
        temperature, top_p: Sampling parameters.
        timeout: HTTP timeout per request.
        max_workers: Concurrent in-flight requests.

    Returns:
        List of generated token-id sequences (one per input), in input order.
        Each sequence contains only the new tokens — no echo of the prompt.

    Raises:
        RuntimeError: If any sample's response is missing the `token_ids` field
            after vLLM said the request succeeded. (We set `return_token_ids=True`
            in the payload, so a server that respects the request always returns
            token_ids on a 2xx response; absence indicates a server version
            mismatch.) Caller can catch and decide whether to skip the batch.
    """
    base_url = base_url.rstrip('/')
    model = _resolve_model_name(base_url)
    url = f'{base_url}/v1/completions'
    batch_size = len(prompt_token_ids_list)
    if isinstance(max_tokens, int):
        per_sample_max = [max_tokens] * batch_size
    else:
        per_sample_max = list(max_tokens)
        if len(per_sample_max) != batch_size:
            raise ValueError(
                f'max_tokens list length {len(per_sample_max)} != batch_size {batch_size}'
            )
    outputs: List[List[int]] = [[] for _ in range(batch_size)]
    errors: List[Optional[str]] = [None] * batch_size

    def _fetch_one(idx: int) -> None:
        if per_sample_max[idx] <= 0:
            # Nothing to generate (suffix budget exhausted by teacher prefix).
            return
        payload = {
            'model': model,
            'prompt': prompt_token_ids_list[idx],
            'max_tokens': per_sample_max[idx],
            'temperature': temperature,
            'top_p': top_p,
            'echo': False,
            'return_token_ids': True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            choice = resp.json()['choices'][0]
            token_ids = choice.get('token_ids')
            if token_ids is None:
                errors[idx] = (
                    f'choices[0].token_ids missing despite return_token_ids=True; '
                    f'response keys={list(choice.keys())}'
                )
                return
            outputs[idx] = list(token_ids)
        except Exception as e:
            errors[idx] = f'{type(e).__name__}: {e}'

    if batch_size > 0:
        with ThreadPoolExecutor(max_workers=min(batch_size, max_workers)) as pool:
            list(pool.map(_fetch_one, range(batch_size)))

    failed = [i for i, err in enumerate(errors) if err is not None]
    if failed:
        msg = '; '.join(f'sample {i}: {errors[i]}' for i in failed[:5])
        more = f' (and {len(failed) - 5} more)' if len(failed) > 5 else ''
        raise RuntimeError(f'vllm_complete: {len(failed)}/{batch_size} samples failed: {msg}{more}')

    return outputs
