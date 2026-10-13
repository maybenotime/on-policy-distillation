# verl_opd — entropy-shape / teacher-selection OPD experiments (verl 0.9.0, SFU Solar)

Code behind the "shape of uncertainty" OPD experiments: k1+PG on-policy distillation in verl with per-step
entropy-shape probes (ESC ρ, Overlap@16, ΔH, ForkOverlap), the entropy-shape transplant (synthetic teacher
A ⊕ shape(B)), correctness-controlled subsets, teacher-outcome prediction, and the paper figures.
Everything ran on SFU Solar `cs-venus-05` / `cs-venus-19` (Blackwell L40S-class, 96 GB); paths are
cluster-specific (`/home/sya225/opd`, `/localscratch/sya225`) and are set in `env.sh`.

Layout (mirrors the on-cluster tree `/home/sya225/opd/`, `/home/sya225/slurm/`, `/home/sya225/*.py`):

| dir | what |
|---|---|
| `env.sh` | portable env: `$OPD_HOME`, `$SCRATCH`, venvs (`envs/verl_opd`, `envs/code_eval_env`), HF cache |
| `plugins/k1_probe/` | verl plugin: `k1_probe.py` (k1+PG loss + per-step probes `probe/rho`, `probe/overlap16`, `probe/dH`, `probe/fork_overlap`), `fsdp_k1_probe.py` (teacher keeps the sampled-token log-prob, `VERL_TEACHER_KEEP_SAMPLED=1`), `token_select.py`, `losses.patch` / `utils.patch` (applied to verl by `scripts/patch_losses.py`, `patch_utils.py`) |
| `plugins/k1_probe/tp_stage/` | entropy-shape transplant: `tp_core.py` (per-position temperature solver, Newton in log T), `k1_transplant_loss.py` (`DISTILLATION_LOSS_MODE=k1_transplant`, `TRANSPLANT_MODE=shape|student|flat|uniform`), `fsdp_k1_probe_tp_patch.py`, `chain19_tp.sh` launcher |
| `scripts/` | training chains (`chain05_*.sh`, `chain19_*.sh`, `chain19_tp.sh`, `run_opd_math.sh`), MathArena eval (`run_matharena.sh`, `eval_*.sh`, `ma_split.sh`, `agg_math.py`), rollout + probe pipeline (`fork_rollout.py` → `fork_entropy.py` / `fork_topk.py` → `fork_metrics.py`, `overlap_metrics.py`, `rho_*.py`), teacher screening (`screen_teachers.sh`, `teacher_cmp.py`), hidden-state probe (`hidden_probe.py`), DeepMath subsets (`deepmath_*.py`), transplant diagnostics (`tp_diag.py`, `tp_uniformT.py`, `extract_tp.py`), log → csv (`log2csv*.py`) |
| `slurm/` | SLURM drivers used for every run (`driver05*.sh`, `driver19*.sh`, `hold_venus*.sh` node holders, `teacher_lane*.sh`, `probe_*_pfo*.sh` fixed-vs-on-policy probes, `teaser_big.sh`) |
| `solar_home/` | small helpers that lived in `~` on Solar: `status_score.py` (score MathArena status files), `signflip.py` (temperature sign-flip measurement), `teaser_data*.py` / `teaser_scan_big.py` / `teaser_dump_full.py` (Figure 1 data), `series*.py`, `ma_score.py`, `ma_merge.py`, `k1*.py` |
| `analysis/` | local plotting / table scripts: `plot_dynamics_2x4.py` (Fig. 2), `plot_fig1b_v2.py`, `plot_fig1c_v2.py`, `plot_entropy_shape_v2.py`, `plot_base_grpo.py`, `plot_run4.py`, `plot_ds_shape.py`, `plot_tp_curves.py`, `teaser3.py` / `teaser3_col.py` (Fig. 1), `predict_table*.py`, `mean6.py` |

Training recipe (all runs): k1+PG, token-mean, AdamW lr 1e-6 const, wd 0.01, grad-clip 1.0, batch = mini-batch 64,
rollout n = 1, T 1.0 / top-p 1.0, prompt 1024 / response 16384, DAPO-Math-17k (17,398 prompts, 200 steps ≈ 0.74 epoch)
or DeepMath 7k subsets (109 steps = 1 epoch). Eval: MathArena, T 0.7 / top-p 0.8 / top-k 20, n = 4
(DeepSeek 16k, Qwen3 8k), six competitions (AIME-24 I+II merged, AIME-25, AMC, BeyondAIME, HMMT-25, HMMT-26).

Not included: venvs, the verl checkout, model weights, rollouts / entropy dumps (`opd/fork/*.npz|jsonl`),
MathArena outputs, and the probe CSVs (those live in the experiment log repo under `results/`).
