"""
Visualize per-token rewards and entropy from token_reward_analysis.py output.

For each sample produces:
  - Completion text where each token's background color encodes the reward
    difference across teachers: white = teachers agree, deep purple = large gap.
    With 2 teachers: |reward_A - reward_B|.
    With >2 teachers: max(rewards) - min(rewards) across all teachers.
    Hover tooltip shows: token text, entropy, and each teacher's reward.
  - A line chart showing reward (log p_teacher - log p_student) per teacher
    overlaid with a shaded entropy curve.

Output: a single self-contained HTML file.

Usage:
    python visualize_token_rewards.py \
        --input  results/token_rewards_test.pkl \
        --output results/token_rewards_viz.html \
        --model  Qwen/Qwen2.5-0.5B-Instruct   # student tokenizer for decoding
        --max_samples 10                        # optional cap
"""

import argparse
import base64
import html as html_lib
import io
import pickle
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEACHER_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]


def reward_diff_to_hex(diff_val: float, vmax: float) -> str:
    """Map a scalar reward-diff (≥0) to a CSS color (white → deep purple)."""
    cmap = plt.get_cmap("Purples")
    normed = diff_val / (vmax + 1e-9)
    r, g, b, _ = cmap(float(np.clip(normed, 0.0, 1.0)))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def compute_reward_diff(teacher_rewards: dict) -> np.ndarray:
    """Per-token reward spread across teachers (always ≥ 0)."""
    stacked = np.stack(list(teacher_rewards.values()), axis=0)  # (n_teachers, T)
    if stacked.shape[0] == 2:
        return np.abs(stacked[0] - stacked[1])
    return stacked.max(axis=0) - stacked.min(axis=0)


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def decode_tokens(tokenizer, token_ids: list[int]) -> list[str]:
    """Decode each token ID individually, preserving whitespace."""
    return [tokenizer.decode([tid]) for tid in token_ids]


# ---------------------------------------------------------------------------
# Per-sample HTML block
# ---------------------------------------------------------------------------

def render_sample(
    idx: int,
    record: dict,
    tokenizer,
    teacher_names: list[str],
    global_diff_max: float,
) -> str:
    completion_ids = record["completion_token_ids"]
    student_entropy = record["student_entropy"].float().numpy()
    student_logps   = record["student_logps"].float().numpy()
    teacher_rewards = {k: v.float().numpy() for k, v in record["teacher_rewards"].items()}

    token_strings = decode_tokens(tokenizer, completion_ids)
    T = len(token_strings)

    reward_diffs = compute_reward_diff(teacher_rewards)  # shape (T,)

    # ---- colored token text ----
    token_spans = []
    for t, tok_str in enumerate(token_strings):
        diff = float(reward_diffs[t])
        ent  = float(student_entropy[t])
        bg = reward_diff_to_hex(diff, global_diff_max)
        r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#111" if lum > 128 else "#eee"
        escaped = html_lib.escape(tok_str).replace(" ", "&nbsp;").replace("\n", "↵<br>")
        tooltip = (
            f"token #{t}: {html_lib.escape(repr(tok_str))}\\n"
            f"reward diff: {diff:.4f}\\n"
            f"entropy: {ent:.4f}\\n"
        )
        for tname, rew in teacher_rewards.items():
            short = tname.split("/")[-1]
            tooltip += f"reward ({short}): {float(rew[t]):.4f}\\n"
        # Red border when teachers disagree on sign (some positive, some negative)
        rewards_at_t = [float(rew[t]) for rew in teacher_rewards.values()]
        sign_conflict = any(r > 0 for r in rewards_at_t) and any(r < 0 for r in rewards_at_t)
        border = "border:2px solid #e00;box-sizing:border-box;" if sign_conflict else "border:2px solid transparent;"
        span = (
            f'<span style="background:{bg};color:{fg};padding:1px 2px;'
            f'border-radius:2px;font-family:monospace;font-size:13px;'
            f'white-space:pre-wrap;cursor:default;{border}" title="{tooltip}">'
            f"{escaped}</span>"
        )
        token_spans.append(span)

    token_html = "".join(token_spans)

    # ---- reward + entropy line chart ----
    fig, ax1 = plt.subplots(figsize=(12, 3))
    xs = np.arange(T)

    # entropy on secondary axis (filled area)
    ax2 = ax1.twinx()
    ax2.fill_between(xs, student_entropy, alpha=0.12, color="gray", label="entropy")
    ax2.set_ylabel("entropy", color="gray", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=8)
    ax2.set_ylim(bottom=0)

    # reward per teacher on primary axis
    for i, tname in enumerate(teacher_names):
        rew = teacher_rewards[tname]
        color = TEACHER_COLORS[i % len(TEACHER_COLORS)]
        short = tname.split("/")[-1]
        ax1.plot(xs, rew, color=color, linewidth=0.8, alpha=0.85, label=short)

    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax1.set_xlabel("token position", fontsize=9)
    ax1.set_ylabel("reward  (log p_teacher − log p_student)", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()

    chart_b64 = fig_to_base64(fig)

    prompt_escaped = html_lib.escape(record["prompt"])
    is_correct = record.get("is_correct")
    if is_correct is True:
        badge = '<span class="badge correct">✓ Correct</span>'
    elif is_correct is False:
        badge = '<span class="badge incorrect">✗ Incorrect</span>'
    else:
        badge = '<span class="badge unknown">? N/A</span>'

    return f"""
<div class="sample" id="sample-{idx}">
  <h3>Sample {idx} {badge}</h3>
  <details>
    <summary><b>Prompt</b></summary>
    <pre class="prompt">{prompt_escaped}</pre>
  </details>
  <div class="legend-bar">
    <span style="font-size:12px;color:#555;">Teacher reward diff →&nbsp;</span>
    <canvas id="grad-{idx}" width="200" height="16"
      style="border-radius:3px;vertical-align:middle;"></canvas>
    <span style="font-size:11px;color:#555;">&nbsp;small&nbsp;←&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp;large</span>
  </div>
  <script>
  (function(){{
    var c = document.getElementById('grad-{idx}');
    var ctx = c.getContext('2d');
    var grad = ctx.createLinearGradient(0,0,200,0);
    grad.addColorStop(0,'#fcfbfd');
    grad.addColorStop(0.5,'#9e9ac8');
    grad.addColorStop(1,'#3f007d');
    ctx.fillStyle = grad;
    ctx.fillRect(0,0,200,16);
  }})();
  </script>
  <div class="token-text">{token_html}</div>
  <img src="data:image/png;base64,{chart_b64}"
       style="width:100%;margin-top:10px;" alt="reward chart">
  <hr>
</div>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/token_rewards_test.pkl")
    parser.add_argument("--output", default="results/token_rewards_viz.html")
    parser.add_argument("--model",  default=None,
                        help="Student model name/path for tokenizer (defaults to metadata value)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Simple head-N cut (ignored when --sample_n is set)")
    parser.add_argument("--sample_n", type=int, default=None,
                        help="Randomly sample N correct + N incorrect records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    metadata = data["metadata"]
    records  = data["records"]

    if args.sample_n is not None:
        rng = random.Random(args.seed)
        correct_pool   = [r for r in records if r.get("is_correct") is True]
        incorrect_pool = [r for r in records if r.get("is_correct") is False]
        n = args.sample_n
        sampled_correct   = rng.sample(correct_pool,   min(n, len(correct_pool)))
        sampled_incorrect = rng.sample(incorrect_pool, min(n, len(incorrect_pool)))
        records = sampled_correct + sampled_incorrect
        print(f"Sampled {len(sampled_correct)} correct + {len(sampled_incorrect)} incorrect records")
    elif args.max_samples:
        records = records[: args.max_samples]

    model_name    = args.model or metadata["student_model"]
    teacher_names = metadata["teacher_models"]

    print(f"Loading tokenizer from {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Compute global reward-diff max for consistent coloring across samples
    all_diffs = np.concatenate([
        compute_reward_diff({k: v.float().numpy() for k, v in r["teacher_rewards"].items()})
        for r in records
    ])
    global_diff_max = float(all_diffs.max())
    print(f"Global reward-diff max: {global_diff_max:.3f}")

    # Build HTML — split into correct / incorrect groups
    correct_blocks   = []
    incorrect_blocks = []
    other_blocks     = []
    for i, rec in enumerate(records):
        print(f"  Rendering sample {i}/{len(records)-1} …")
        block = render_sample(i, rec, tokenizer, teacher_names, global_diff_max)
        is_correct = rec.get("is_correct")
        if is_correct is True:
            correct_blocks.append(block)
        elif is_correct is False:
            incorrect_blocks.append(block)
        else:
            other_blocks.append(block)

    teacher_legend_items = "".join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{TEACHER_COLORS[i % len(TEACHER_COLORS)]};'
        f'border-radius:2px;margin-right:4px;vertical-align:middle;"></span>'
        f'<span style="margin-right:16px;font-size:13px;">{html_lib.escape(n.split("/")[-1])}</span>'
        for i, n in enumerate(teacher_names)
    )

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Token Reward Visualization</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background:#fafafa; }}
  h1   {{ font-size: 1.4em; color: #333; }}
  h3   {{ margin-bottom: 6px; color: #444; }}
  pre.prompt {{ background:#f0f0f0; padding:8px; border-radius:4px; white-space:pre-wrap;
                font-size:12px; max-height:200px; overflow-y:auto; }}
  .sample  {{ background:#fff; border:1px solid #ddd; border-radius:6px;
              padding:16px; margin-bottom:24px; box-shadow:0 1px 3px rgba(0,0,0,.07); }}
  .token-text {{ line-height: 1.9; padding: 10px; background:#f9f9f9;
                 border:1px solid #eee; border-radius:4px; margin: 8px 0; word-break:break-word; }}
  .legend-bar {{ margin: 6px 0 4px; }}
  details summary {{ cursor:pointer; color:#555; font-size:13px; }}
  hr {{ border:none; border-top:1px solid #eee; }}
  .meta {{ font-size: 12px; color: #666; margin-bottom: 16px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px;
            font-size:12px; font-weight:600; margin-left:8px; vertical-align:middle; }}
  .badge.correct   {{ background:#d4edda; color:#155724; }}
  .badge.incorrect {{ background:#f8d7da; color:#721c24; }}
  .badge.unknown   {{ background:#e2e3e5; color:#383d41; }}
  .group-header {{ font-size:1.15em; font-weight:700; padding:10px 14px;
                   border-radius:6px; margin: 28px 0 12px; }}
  .correct-header   {{ background:#d4edda; color:#155724; }}
  .incorrect-header {{ background:#f8d7da; color:#721c24; }}
  .group-count {{ font-weight:400; font-size:0.9em; }}
</style>
</head>
<body>
<h1>Token Reward Analysis</h1>
<div class="meta">
  <b>Student:</b> {html_lib.escape(metadata["student_model"])} &nbsp;|&nbsp;
  <b>Temperature:</b> {metadata["temperature"]} &nbsp;|&nbsp;
  <b>Max new tokens:</b> {metadata["max_new_tokens"]} &nbsp;|&nbsp;
  <b>Samples:</b> {len(records)}
  <br><br>
  <b>Teacher reward colors:</b>&nbsp; {teacher_legend_items}
  <br>
  <b>Token background:</b> reward difference across teachers at each token
  (white = teachers agree &nbsp;→&nbsp; deep purple = large disagreement);
  hover each token for exact entropy, rewards, and diff values
</div>

{"" if not correct_blocks else f'<div class="group-header correct-header">✓ Correct &nbsp;<span class="group-count">({len(correct_blocks)} samples)</span></div>' + "".join(correct_blocks)}
{"" if not incorrect_blocks else f'<div class="group-header incorrect-header">✗ Incorrect &nbsp;<span class="group-count">({len(incorrect_blocks)} samples)</span></div>' + "".join(incorrect_blocks)}
{"" if not other_blocks else f'<div class="group-header">? No label &nbsp;<span class="group-count">({len(other_blocks)} samples)</span></div>' + "".join(other_blocks)}
</body>
</html>
"""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(html_page, encoding="utf-8")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
