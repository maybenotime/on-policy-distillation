# GKD On-Policy Distillation — Experiment Results

**Setup**: GKD (`swift rlhf --rlhf_type gkd`), dataset `prompts_math_swift.jsonl`, datasize=5000, 1 epoch  
**Student training**: DeepSpeed ZeRO-2, batch size 1, grad accum 10, max_length 8192, max_completion_length 4096, gradient_checkpointing  
**Cluster**: SFU Solar (L40S 44GB × 4) for Exp1/2; Fir (H100 80GB × 4) for Exp3/4/5

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
- **Job**: Fir 33076515 | **Status**: Running (submitted 2026-04-07)

### Exp4: 3B student + 7B teacher
- **Job**: Fir 33076516 | **Status**: Running (submitted 2026-04-07)

### Exp5: 3B student + 14B teacher
- **Job**: Fir 33076517 | **Status**: Running (submitted 2026-04-07)

---

## Summary Table (aggregated scores)

> math_500 score = macro average across all 5 levels  
> amc score = macro average across amc22/23/24  
> aime25 score = macro average across I and II

| Student | Teacher | gsm8k | aime24 | aime25 | amc (avg) | math_500 (avg) |
|---------|---------|------:|-------:|-------:|----------:|---------------:|
| 0.5B | 3B | 0.4215 | 0.0000 | 0.0000 | 0.0519 | 0.3159 |
| 0.5B | 7B | 0.4132 | 0.0000 | 0.0000 | 0.0895 | 0.2977 |
| 0.5B | 14B | — | — | — | — | — |
| 3B | 7B | — | — | — | — | — |
| 3B | 14B | — | — | — | — | — |

---

*Last updated: 2026-04-07. Exp3/4/5 results pending (Fir jobs 33076515/16/17).*
