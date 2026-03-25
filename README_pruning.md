# UniPoRA TA-Prompt Token Pruning

## Scope

This pipeline only prunes TA-Prompt tokens. It does not prune layers, GA-LoRA weights, or the backbone. Pruning is always done per task and per stage: each stage sorts its own prompt tokens and removes the lowest-scoring fraction while keeping at least `MIN_TOKENS_PER_LAYER`.

The implementation is driven by `run_unipora_pruning.py` plus the stage YAMLs under `configs/mtlora/tiny_448/pascal/`.

## Start From A Trained UniPoRA Checkpoint

Use the same dataset path and task list that were used to train the checkpoint. The pruning script reads the trained checkpoint from `--resume` and treats it as the full teacher checkpoint unless `PRUNING.RECOVERY.TEACHER_CHECKPOINT` is overridden.

Example:

```bash
python run_unipora_pruning.py \
  --cfg configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p10.yaml \
  --pascal PASCAL_MT \
  --tasks semseg,normals,sal,human_parts \
  --batch-size 12 \
  --resume /path/to/unipora_teacher_checkpoint.pth
```

## Recommended Run Order

Stage 1 first:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p5.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p10.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage1_base_p15.yaml`

If stage 1 shows that around 10% pruning is stable, run stage 2:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage2_base_recover_p10.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage2_base_recover_p20.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage2_base_recover_p30.yaml`

Then run stage 3 for a fair comparison against stage 2:

1. `configs/mtlora/tiny_448/pascal/unipora_prune_stage3_ga_recover_p10.yaml`
2. `configs/mtlora/tiny_448/pascal/unipora_prune_stage3_ga_recover_p20.yaml`
3. `configs/mtlora/tiny_448/pascal/unipora_prune_stage3_ga_recover_p30.yaml`

## What Each Stage Does

Stage 1:

- computes `S_base`
- applies per-layer prompt pruning
- evaluates immediately without recovery

Stage 2:

- computes `S_base`
- applies per-layer prompt pruning
- runs 5 recovery epochs by default
- freezes backbone and GA-LoRA by default
- trains kept TA-Prompt plus task heads

Stage 3:

- computes `S_GA = S_base * clamp_min(DeltaL_P, 0) / (clamp_min(DeltaL_GA, 0) + eps)`
- keeps the same pruning and recovery recipe as stage 2
- changes only the importance score

Where:

- `S_base(t,l,k) = |dL_t / dm[t,l,k]|`
- `DeltaL_P(t,l,k) = L_t(drop only token k at layer l) - L_t(full)`
- `DeltaL_GA(t,l) = L_t(drop GA-LoRA at layer l) - L_t(full)`

## Output Layout

Each run writes to:

`output/<model_name>/<tag>/pruning/<experiment_name>/`

The directory contains:

- `importance_scores.pth`
- `importance_scores.json`
- `pruning_mask.pth`
- `pruning_mask.json`
- `pruning_summary.json`
- `pruned_checkpoint.pth`
- `eval_before_recovery.json`
- `recovery_checkpoint.pth`
- `eval_after_recovery.json`
- `experiment_summary.json`

`pruning_summary.json` now stores the full pruning payload, including:

- `score_type`
- `prune_ratio`
- `min_tokens_per_layer`
- `keep_indices`
- `summary`
- `prompt_statistics`
- `aggregate`

The `aggregate` section includes:

- `total_original_tokens`
- `total_kept_tokens`
- `total_pruned_tokens`
- `overall_keep_ratio`
- `overall_prune_ratio`

## Reuse Modes

The YAML flags support partial reruns:

- `PRUNING.COMPUTE_IMPORTANCE`
- `PRUNING.APPLY_PRUNING`
- `PRUNING.RUN_RECOVERY`
- `PRUNING.EVALUATE_AFTER_PRUNING`
- `PRUNING.IMPORTANCE.LOAD_PATH`
- `PRUNING.PRUNER.LOAD_MASK`
- `PRUNING.PRUNER.LOAD_PRUNED_CHECKPOINT`
- `PRUNING.RECOVERY.STUDENT_CHECKPOINT`

If you have an older stage 3 importance file that was generated before `token_delta_losses` was added, you must recompute stage 3 importance. The new loader will reject old stage 3 importance payloads instead of silently reusing them.

## Result Aggregation

Aggregate completed experiments with:

```bash
python scripts/summarize_pruning_results.py \
  --results-root output/promlora_tiny_448_r64_prom50_scale4_pertask/unipora_ta_prompt_pruning/pruning \
  --csv-out output/pruning_summary.csv \
  --md-out output/pruning_summary.md
```
