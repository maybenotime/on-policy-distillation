"""Wrapper around `swift rollout` that adds raw `prompt_token_ids` prefill.

ms-swift's rollout server only accepts `messages` (no `prompt_token_ids`),
which breaks TP-BPOD's prefix-injection design. This wrapper monkey-patches
`Template.encode` to short-circuit when the request's `data_dict` carries
a `prompt_token_ids` field — in that case it skips chat-template encoding
and feeds the raw token IDs straight to vLLM. The unchanged path is fully
preserved for all other requests (e.g., the existing SP-FPOD on-policy
rollout that goes through messages).

Usage: invoked from `scripts/rollout.sh` when PREFIX_DIRECTION=backward.
Forwards all CLI args to `swift rollout` via runpy.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _install_prompt_token_ids_patch() -> None:
    import sys
    from swift.template.base import Template
    from swift.template.template_inputs import TemplateInputs

    _orig_encode = Template.encode

    def _patched_encode(
        self,
        inputs,
        return_template_inputs: bool = False,
        return_length: bool = False,
    ) -> Dict[str, Any]:
        # Debug stderr so we know the patch is on the inference path.
        print('[PATCHED_ENCODE_CALLED] inputs_type=', type(inputs).__name__,
              'has_data_dict=', hasattr(inputs, 'data_dict'),
              flush=True, file=sys.stderr)
        data_dict = getattr(inputs, 'data_dict', None)
        token_ids: Optional[list] = None
        if isinstance(data_dict, dict):
            token_ids = data_dict.get('prompt_token_ids')

        if token_ids:
            print('[PATCHED_ENCODE_HIT] using prompt_token_ids len=', len(token_ids),
                  flush=True, file=sys.stderr)
            ids = list(token_ids)
            encoded: Dict[str, Any] = {
                'input_ids': ids,
                'attention_mask': [1] * len(ids),
                'labels': None,
            }
            if return_length:
                encoded['length'] = len(ids)
            if return_template_inputs:
                encoded['template_inputs'] = TemplateInputs.from_dict(
                    {'messages': [{'role': 'user', 'content': ''}]}
                )
            return encoded

        return _orig_encode(
            self,
            inputs,
            return_template_inputs=return_template_inputs,
            return_length=return_length,
        )

    Template.encode = _patched_encode
    print('[rollout_with_prefill] patched Template.encode (id=', id(Template), ')',
          flush=True, file=sys.stderr)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    _install_prompt_token_ids_patch()

    import runpy
    runpy.run_module('swift.cli.rollout', run_name='__main__')


if __name__ == '__main__':
    main()
