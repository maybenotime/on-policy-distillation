#!/bin/bash
# Late-Intervention OPSD launcher.
#
# This entrypoint is intentionally fail-fast until LI-OPSD is implemented in the
# trainer/plugin layer. Running ordinary opsd.sh with LI_OPSD_RATIO set would
# silently reproduce OPSD, not late-intervention OPSD.

set -euo pipefail

echo "LI_OPSD_RATIO=${LI_OPSD_RATIO:-unset}"
echo "ERROR: LI-OPSD requires trainer/plugin support for suffix-only KL masking." >&2
echo "Do not run this launcher until source/li_opsd_plugin.py or an equivalent" >&2
echo "GKD trainer patch implements LI_OPSD_RATIO." >&2
exit 2
