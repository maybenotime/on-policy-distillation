"""swift rollout `/infer/` client for TP-BPOD.

ms-swift's rollout server (started by `swift rollout`) exposes only a custom
`/infer/` endpoint, not vLLM's stock `/v1/completions`. To inject a raw
prompt-token sequence we ride on the `RolloutInferRequest.data_dict` field as
a carrier — a sibling wrapper script (`rollout_with_prefill.py`) patches
`Template.encode` server-side to honor `data_dict['prompt_token_ids']` and
skip chat-template encoding.

Returns generated token IDs (the student-written suffix only — vLLM does
not echo the prompt+prefix block).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)


def vllm_complete(
    base_url: str,
    prompt_token_ids_list: List[List[int]],
    max_tokens: Union[int, Sequence[int]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    timeout: float = 600.0,
    max_workers: int = 8,
) -> List[List[int]]:
    """POST `prompt_token_ids` to swift rollout `/infer/`; return generated tokens.

    Args:
        base_url: Server base URL, e.g. 'http://127.0.0.1:8000'.
        prompt_token_ids_list: One list of token IDs per sample.
        max_tokens: Single int or per-sample sequence.
        temperature, top_p: Sampling parameters.
        timeout: HTTP timeout per request.
        max_workers: Concurrent in-flight requests.

    Returns:
        List of generated-token sequences (one per input). Empty for samples
        whose request failed (caller decides; we raise if any failed).

    Raises:
        RuntimeError: If any sample's response is missing token IDs.
    """
    base_url = base_url.rstrip('/')
    url = f'{base_url}/infer/'
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
            return
        # Carry prompt_token_ids via data_dict; the server-side
        # rollout_with_prefill patch unpacks them and skips template.encode.
        # We still send a dummy `messages` field because RolloutInferRequest
        # requires it; the patched encoder ignores it.
        infer_request = {
            'messages': [{'role': 'user', 'content': ''}],
            'data_dict': {'prompt_token_ids': prompt_token_ids_list[idx]},
        }
        request_config = {
            'max_tokens': per_sample_max[idx],
            'temperature': temperature,
            'top_p': top_p,
            'return_details': True,
        }
        payload = {
            'infer_requests': [infer_request],
            'request_config': request_config,
        }
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or not data:
                errors[idx] = f'unexpected response shape: {type(data).__name__}'
                return
            entry = data[0]
            # Preferred: response_token_ids (top-level field on the entry)
            tids = entry.get('response_token_ids')
            if not tids:
                # Fallback: choices[0].token_ids inside the response
                choice = entry.get('response', {}).get('choices', [{}])[0]
                tids = choice.get('token_ids')
            if not tids:
                errors[idx] = (
                    f'no token_ids in response; entry keys={list(entry.keys())}, '
                    f"choice keys={list(entry.get('response', {}).get('choices', [{}])[0].keys())}"
                )
                return
            outputs[idx] = list(tids)
        except Exception as e:
            errors[idx] = f'{type(e).__name__}: {e}'

    if batch_size > 0:
        with ThreadPoolExecutor(max_workers=min(batch_size, max_workers)) as pool:
            list(pool.map(_fetch_one, range(batch_size)))

    failed = [i for i, err in enumerate(errors) if err is not None]
    if failed:
        msg = '; '.join(f'sample {i}: {errors[i]}' for i in failed[:5])
        more = f' (and {len(failed) - 5} more)' if len(failed) > 5 else ''
        raise RuntimeError(
            f'vllm_complete: {len(failed)}/{batch_size} samples failed: {msg}{more}'
        )

    return outputs
