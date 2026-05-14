# GKD On-Policy Distillation — Experiment Results

**Setup**: GKD (`swift rlhf --rlhf_type gkd`), dataset `prompts_math_swift.jsonl`, datasize=5000, 1 epoch  
**Student training**: DeepSpeed ZeRO-2, batch size 1, grad accum 10, max_length 8192, max_completion_length 4096, gradient_checkpointing  
**Cluster**: SFU Solar (L40S 44GB × 4) for Exp1/2; Fir (H100 80GB × 4) for Exp3/4/5

---

## Baseline: Qwen2.5-3B-Instruct (no distillation)
- **Job**: Fir 34347364

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.8544 |
| aime24 | default (30) | 0.0667 |
| aime25 | AIME2025-I (15) | 0.0667 |
| aime25 | AIME2025-II (15) | 0.0667 |
| amc | amc22 (43) | 0.3256 |
| amc | amc23 (46) | 0.4130 |
| amc | amc24 (45) | 0.2889 |
| math_500 | Level 1 (43) | 0.8605 |
| math_500 | Level 2 (90) | 0.8222 |
| math_500 | Level 3 (105) | 0.8095 |
| math_500 | Level 4 (128) | 0.6016 |
| math_500 | Level 5 (134) | 0.4030 |

---

## Baseline: Qwen2.5-0.5B-Instruct (no distillation)

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.4466 |
| aime24 | default (30) | 0.0000 |
| aime25 | AIME2025-I (15) | 0.0000 |
| aime25 | AIME2025-II (15) | 0.0000 |
| amc | amc22 (43) | 0.0698 |
| amc | amc23 (46) | 0.1304 |
| amc | amc24 (45) | 0.0444 |
| math_500 | Level 1 (43) | 0.7674 |
| math_500 | Level 2 (90) | 0.4333 |
| math_500 | Level 3 (105) | 0.4286 |
| math_500 | Level 4 (128) | 0.1797 |
| math_500 | Level 5 (134) | 0.0597 |

---

## Results

### Exp1: 0.5B student + 3B teacher
- **Job**: Solar 8923 | **Checkpoint**: `output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-3B-Instruct_data5000/checkpoint-417`

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.4215 |
| aime24 | default (30) | 0.0000 |
| aime25 | AIME2025-I (15) | 0.0000 |
| aime25 | AIME2025-II (15) | 0.0000 |
| amc | amc22 (43) | 0.0465 |
| amc | amc23 (46) | 0.0870 |
| amc | amc24 (45) | 0.0222 |
| math_500 | Level 1 (43) | 0.5814 |
| math_500 | Level 2 (90) | 0.4000 |
| math_500 | Level 3 (105) | 0.3524 |
| math_500 | Level 4 (128) | 0.1562 |
| math_500 | Level 5 (134) | 0.0896 |

### Exp2: 0.5B student + 7B teacher
- **Job**: Solar 8917 | **Checkpoint**: `output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-7B-Instruct_data5000/checkpoint-417`

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.4132 |
| aime24 | default (30) | 0.0000 |
| aime25 | AIME2025-I (15) | 0.0000 |
| aime25 | AIME2025-II (15) | 0.0000 |
| amc | amc22 (43) | 0.1163 |
| amc | amc23 (46) | 0.1522 |
| amc | amc24 (45) | 0.0000 |
| math_500 | Level 1 (43) | 0.6047 |
| math_500 | Level 2 (90) | 0.4222 |
| math_500 | Level 3 (105) | 0.2762 |
| math_500 | Level 4 (128) | 0.1406 |
| math_500 | Level 5 (134) | 0.0448 |

### Exp3: 0.5B student + 14B teacher
- **Job**: Fir 33466692 (eval 34326514) | **Checkpoint**: `output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-14B-Instruct_data5000/v1-20260408-220734/checkpoint-417`

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.4352 |
| aime24 | default (30) | 0.0333 |
| aime25 | AIME2025-I (15) | 0.0000 |
| aime25 | AIME2025-II (15) | 0.0000 |
| amc | amc22 (43) | 0.0465 |
| amc | amc23 (46) | 0.0435 |
| amc | amc24 (45) | 0.0222 |
| math_500 | Level 1 (43) | 0.5349 |
| math_500 | Level 2 (90) | 0.4111 |
| math_500 | Level 3 (105) | 0.2857 |
| math_500 | Level 4 (128) | 0.1562 |
| math_500 | Level 5 (134) | 0.0672 |

### Exp4: 3B student + 7B teacher
- **Job**: Fir 33466693 (eval 34326515) | **Checkpoint**: `output/student_Qwen2.5-3B-Instruct_teacher_Qwen2.5-7B-Instruct_data5000/v1-20260408-220849/checkpoint-417`

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.8052 |
| aime24 | default (30) | 0.0333 |
| aime25 | AIME2025-I (15) | 0.0667 |
| aime25 | AIME2025-II (15) | 0.0000 |
| amc | amc22 (43) | 0.2791 |
| amc | amc23 (46) | 0.2174 |
| amc | amc24 (45) | 0.2000 |
| math_500 | Level 1 (43) | 0.8372 |
| math_500 | Level 2 (90) | 0.8444 |
| math_500 | Level 3 (105) | 0.6571 |
| math_500 | Level 4 (128) | 0.5625 |
| math_500 | Level 5 (134) | 0.2761 |

### Exp5: 3B student + 14B teacher
- **Job**: Fir 33466694 (eval 34326516) | **Checkpoint**: `output/student_Qwen2.5-3B-Instruct_teacher_Qwen2.5-14B-Instruct_data5000/v1-20260408-225401/checkpoint-417`

| Dataset | Subset | Score |
|---------|--------|------:|
| gsm8k | main (1319) | 0.8287 |
| aime24 | default (30) | 0.1000 |
| aime25 | AIME2025-I (15) | 0.0000 |
| aime25 | AIME2025-II (15) | 0.0667 |
| amc | amc22 (43) | 0.2326 |
| amc | amc23 (46) | 0.2609 |
| amc | amc24 (45) | 0.1333 |
| math_500 | Level 1 (43) | 0.8372 |
| math_500 | Level 2 (90) | 0.8111 |
| math_500 | Level 3 (105) | 0.7048 |
| math_500 | Level 4 (128) | 0.5703 |
| math_500 | Level 5 (134) | 0.3060 |

---

## Summary Table (aggregated scores)

> math_500 score = macro average across all 5 levels  
> amc score = macro average across amc22/23/24  
> aime25 score = macro average across I and II

| Student | Teacher | gsm8k | aime24 | aime25 | amc (avg) | math_500 (avg) |
|---------|---------|------:|-------:|-------:|----------:|---------------:|
| **0.5B** | **baseline** | **0.4466** | **0.0000** | **0.0000** | **0.0482** | **0.3737** |
| 0.5B | 3B | 0.4215 | 0.0000 | 0.0000 | 0.0519 | 0.3159 |
| 0.5B | 7B | 0.4132 | 0.0000 | 0.0000 | 0.0895 | 0.2977 |
| 0.5B | 14B | 0.4352 | 0.0333 | 0.0000 | 0.0374 | 0.2910 |
| **3B** | **baseline** | **0.8544** | **0.0667** | **0.0667** | **0.3425** | **0.6994** |
| 3B | 7B | 0.8052 | 0.0333 | 0.0333 | 0.2322 | 0.6355 |
| 3B | 14B | 0.8287 | 0.1000 | 0.0333 | 0.2089 | 0.6459 |

---

*Last updated: 2026-04-12. All experiments complete. 0.5B baseline = Qwen2.5-0.5B-Instruct, 3B baseline = Qwen2.5-3B-Instruct, both without distillation.*
