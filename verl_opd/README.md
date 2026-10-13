# verl_opd — entropy-shape / teacher-selection OPD experiments (verl 0.9.0)

Code behind the "shape of uncertainty" OPD experiments: k1+PG on-policy distillation in verl with per-step
entropy-shape probes (ESC ρ, Overlap@16, ΔH, ForkOverlap), the entropy-shape transplant (synthetic teacher
A ⊕ shape(B)), correctness-controlled subsets, teacher-outcome prediction, and the paper figures.
Everything ran on two SLURM GPU nodes (`node-A`, `node-B`; Blackwell-class, 96 GB); paths are
cluster-specific (`/home/USER/opd`, `/localscratch/USER`) and are set in `env.sh`.

Layout (mirrors the on-cluster tree `/home/USER/opd/`):

| dir | what |
|---|---|
| `env.sh` | portable env: `$OPD_HOME`, `$SCRATCH`, venvs (`envs/verl_opd`, `envs/code_eval_env`), HF cache |
| `plugins/k1_probe/` | verl plugin: `k1_probe.py` (k1+PG loss + per-step probes `probe/rho`, `probe/overlap16`, `probe/dH`, `probe/fork_overlap`), `fsdp_k1_probe.py` (teacher keeps the sampled-token log-prob, `VERL_TEACHER_KEEP_SAMPLED=1`), `token_select.py`, `losses.patch` / `utils.patch` (applied to verl by `scripts/patch_losses.py`, `patch_utils.py`) |
| `plugins/k1_probe/tp_stage/` | entropy-shape transplant: `tp_core.py` (per-position temperature solver, Newton in log T), `k1_transplant_loss.py` (`DISTILLATION_LOSS_MODE=k1_transplant`, `TRANSPLANT_MODE=shape|student|flat|uniform`), `fsdp_k1_probe_tp_patch.py`, `chain19_tp.sh` launcher |
| `scripts/` | training chains (`chain05_*.sh`, `chain19_*.sh`, `chain19_tp.sh`, `run_opd_math.sh`), MathArena eval (`run_matharena.sh`, `eval_*.sh`, `ma_split.sh`, `agg_math.py`), rollout + probe pipeline (`fork_rollout.py` → `fork_entropy.py` / `fork_topk.py` → `fork_metrics.py`, `overlap_metrics.py`, `rho_*.py`), teacher screening (`screen_teachers.sh`, `teacher_cmp.py`), hidden-state probe (`hidden_probe.py`), DeepMath subsets (`deepmath_*.py`), transplant diagnostics (`tp_diag.py`, `tp_uniformT.py`, `extract_tp.py`), log → csv (`log2csv*.py`) |

Training recipe (all runs): k1+PG, token-mean, AdamW lr 1e-6 const, wd 0.01, grad-clip 1.0, batch = mini-batch 64,
rollout n = 1, T 1.0 / top-p 1.0, prompt 1024 / response 16384, DAPO-Math-17k (17,398 prompts, 200 steps ≈ 0.74 epoch)
or DeepMath 7k subsets (109 steps = 1 epoch). Eval: MathArena, T 0.7 / top-p 0.8 / top-k 20, n = 4
(DeepSeek 16k, Qwen3 8k), six competitions (AIME-24 I+II merged, AIME-25, AMC, BeyondAIME, HMMT-25, HMMT-26).

Not included: venvs, the verl checkout, model weights, rollouts / entropy dumps (`opd/fork/*.npz|jsonl`),
MathArena outputs, the probe CSVs, the SLURM driver scripts, and the local plotting scripts.
