import csv
import json
import os
import random
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from yacs.config import CfgNode as CN

from ag_mtlora.config_utils import (
    build_task_to_group,
    canonicalize_groups,
    enumerate_candidate_groups,
    enumerate_partitions,
    resolve_group_shared_ranks,
    select_predictor_train_groups,
)
from data import build_loader
from data.mtl_ds import get_tasks_config
from logger import create_logger
from models import build_model, build_mtl_model
from models.lora import mark_only_lora_as_trainable
from mtl_loss_schemes import MultiTaskLoss, get_loss
from optimizer import build_optimizer
from utils import load_checkpoint, load_pretrained, mkdir_if_missing


DEFAULT_TASK_LOSS_WEIGHTS = {
    "depth": 1.0,
    "semseg": 1.0,
    "human_parts": 2.0,
    "sal": 5.0,
    "edge": 50.0,
    "normals": 10.0,
}


def group_to_key(group: Sequence[str]) -> str:
    return "|".join(group)


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def save_json(payload: Dict, output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_matrix_csv(tasks: Sequence[str], matrix: np.ndarray, output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task"] + list(tasks))
        for task, row in zip(tasks, matrix.tolist()):
            writer.writerow([task] + row)


def save_group_task_rows(rows: Iterable[Dict], output_path: str, value_key: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "task", value_key])
        for row in rows:
            writer.writerow([row["group"], row["task"], row[value_key]])


def save_predictor_value_csv(value_by_group: Dict[str, Dict[str, float]], output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "task", "value"])
        for group_key, group_values in value_by_group.items():
            for task, value in group_values.items():
                writer.writerow([group_key, task, float(value)])


def save_partition_csv(ranked_partitions: List[Dict], output_path: str) -> None:
    mkdir_if_missing(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "partition_score", "num_groups", "groups", "per_task_scores"])
        for rank, item in enumerate(ranked_partitions, start=1):
            writer.writerow(
                [
                    rank,
                    float(item["partition_score"]),
                    int(item["num_groups"]),
                    json.dumps(item["groups"], ensure_ascii=False),
                    json.dumps(item["per_task_scores"], ensure_ascii=False),
                ]
            )


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def rebuild_mtlora_task_lists(config: CN) -> None:
    if not config.MODEL.MTLORA.ENABLED:
        return

    num_stages = len(config.MODEL.SWIN.DEPTHS)
    config.MODEL.MTLORA.R_PER_TASK_LIST = []
    config.MODEL.MTLORA.SCALE_PER_TASK_LIST = []
    for stage_idx in range(num_stages):
        layer_task_r = {
            "shared": (
                config.MODEL.MTLORA.R_PER_TASK["shared"][stage_idx]
                if "shared" in config.MODEL.MTLORA.R_PER_TASK
                else config.MODEL.MTLORA.R[stage_idx]
            )
        }
        layer_task_scale = {}
        for task in config.TASKS:
            layer_task_r[task] = config.MODEL.MTLORA.R_PER_TASK[task][stage_idx]
            layer_task_scale[task] = config.MODEL.MTLORA.SCALE_PER_TASK[task][stage_idx]
        config.MODEL.MTLORA.R_PER_TASK_LIST.append(layer_task_r)
        config.MODEL.MTLORA.SCALE_PER_TASK_LIST.append(layer_task_scale)


def rebuild_task_config(base_config: CN, tasks: Sequence[str], output_dir: str = None) -> CN:
    config = base_config.clone()
    config.defrost()
    config.TASKS = list(tasks)
    config.MTL = True
    task_cfg, _ = get_tasks_config(config.DATA.DBNAME, config.TASKS, config.DATA.IMG_SIZE)
    task_cfg = dict(task_cfg)
    config.TASKS_CONFIG = CN(task_cfg)
    config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT = CN(dict(config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT))
    config.TASKS_CONFIG.ALL_TASKS.FLAGVALS = CN(dict(config.TASKS_CONFIG.ALL_TASKS.FLAGVALS))
    config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS = CN(dict(config.TASKS_CONFIG.ALL_TASKS.INFER_FLAGVALS))

    if config.MODEL.MTLORA.ENABLED:
        filtered_r_per_task = CN(new_allowed=True)
        filtered_scale_per_task = CN(new_allowed=True)
        for task in config.TASKS:
            filtered_r_per_task[task] = list(base_config.MODEL.MTLORA.R_PER_TASK[task])
            filtered_scale_per_task[task] = list(base_config.MODEL.MTLORA.SCALE_PER_TASK[task])
        if "shared" in base_config.MODEL.MTLORA.R_PER_TASK:
            filtered_r_per_task["shared"] = list(base_config.MODEL.MTLORA.R_PER_TASK["shared"])
        config.MODEL.MTLORA.R_PER_TASK = filtered_r_per_task
        config.MODEL.MTLORA.SCALE_PER_TASK = filtered_scale_per_task
        rebuild_mtlora_task_lists(config)

    config.MODEL.AGMTLORA.ENABLED = False
    config.MODEL.MTLORA.AGMTLORA_ENABLED = False
    config.MODEL.MTLORA.AGMTLORA_STAGE = 0
    config.MODEL.MTLORA.AGMTLORA_GROUPS = []
    config.MODEL.MTLORA.AGMTLORA_GROUP_NAMES = []
    config.MODEL.MTLORA.AGMTLORA_GROUP_RANKS = []
    config.MODEL.MTLORA.AGMTLORA_TASK_TO_GROUP = CN(new_allowed=True)

    if output_dir is not None:
        config.OUTPUT = output_dir
    config.freeze()
    return config


def build_loss_bundle(config: CN):
    loss_ft = nn.ModuleDict(
        {task: get_loss(config["TASKS_CONFIG"], task, config) for task in config.TASKS}
    )
    loss_weights = {task: DEFAULT_TASK_LOSS_WEIGHTS[task] for task in config.TASKS}
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)
    return criterion, loss_ft, loss_weights


def move_batch_to_device(batch: Dict, tasks: Sequence[str], device: torch.device):
    samples = batch["image"].to(device, non_blocking=True)
    targets = {task: batch[task].to(device, non_blocking=True) for task in tasks}
    return samples, targets


def maybe_load_initial_weights(config: CN, model: nn.Module, logger) -> None:
    old_eval_mode = bool(config.EVAL_MODE)
    config.defrost()
    config.EVAL_MODE = True
    config.freeze()
    try:
        if config.MODEL.RESUME:
            load_checkpoint(config, model, None, None, None, logger, quiet=True)
        elif config.MODEL.RESUME_BACKBONE:
            target_model = model.backbone if hasattr(model, "backbone") else model
            load_checkpoint(config, target_model, None, None, None, logger, backbone=True, quiet=True)
        elif config.MODEL.PRETRAINED:
            load_pretrained(config, model, logger)
    finally:
        config.defrost()
        config.EVAL_MODE = old_eval_mode
        config.freeze()


def build_task_model(config: CN, device: torch.device, logger, init_state_dict: Dict[str, torch.Tensor] = None):
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)
    else:
        maybe_load_initial_weights(config, model, logger)

    model.to(device)
    if config.MODEL.MTLORA.ENABLED and config.MODEL.MTLORA.FREEZE_PRETRAINED:
        mark_only_lora_as_trainable(
            model.backbone,
            bias=config.MODEL.MTLORA.BIAS,
            freeze_patch_embed=config.TRAIN.FREEZE_PATCH_EMBED,
            freeze_norm=config.TRAIN.FREEZE_LAYER_NORM,
            free_relative_bias=config.TRAIN.FREEZE_RELATIVE_POSITION_BIAS,
            freeze_downsample_reduction=(
                True if config.MODEL.MTLORA.DOWNSAMPLER_ENABLED else config.TRAIN.FREEZE_DOWNSAMPLE_REDUCTION
            ),
        )
    return model


def evaluate_task_losses(model, data_loader, loss_ft, tasks, device: torch.device, max_batches: int = None):
    model.eval()
    aggregated = {task: 0.0 for task in tasks}
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            samples, targets = move_batch_to_device(batch, tasks, device)
            outputs = model(samples)
            for task in tasks:
                aggregated[task] += float(loss_ft[task](outputs[task], targets[task]).item())
            num_batches += 1
    if num_batches == 0:
        return {task: 0.0 for task in tasks}
    return {task: value / float(num_batches) for task, value in aggregated.items()}


def short_train_model(
    config: CN,
    logger,
    init_state_dict: Dict[str, torch.Tensor],
    num_epochs: int,
    device: torch.device,
):
    dataset_train, dataset_val, data_loader_train, data_loader_val, _ = build_loader(config)
    model = build_task_model(config, device, logger, init_state_dict=init_state_dict)
    criterion, loss_ft, _ = build_loss_bundle(config)
    optimizer = build_optimizer(config, model)

    model.train()
    for epoch_idx in range(int(num_epochs)):
        for batch in data_loader_train:
            samples, targets = move_batch_to_device(batch, config.TASKS, device)
            outputs = model(samples)
            loss, _ = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    val_losses = evaluate_task_losses(model, data_loader_val, loss_ft, config.TASKS, device)
    trained_state = clone_state_dict_to_cpu(model)
    del dataset_train, dataset_val, data_loader_train, data_loader_val
    del optimizer, criterion, loss_ft, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return trained_state, val_losses


def get_shared_ta_parameters(model: nn.Module):
    params = []
    for name, param in model.named_parameters():
        if "backbone" not in name:
            continue
        if "lora_shared_" not in name:
            continue
        if "groups" in name:
            continue
        params.append(param)
    return params


def flatten_gradient_list(params, grads):
    flattened = []
    for param, grad in zip(params, grads):
        if grad is None:
            flattened.append(torch.zeros_like(param).reshape(-1))
        else:
            flattened.append(grad.detach().reshape(-1))
    return torch.cat(flattened, dim=0)


def build_pseudo_update(flat_grad: torch.Tensor, optimizer) -> torch.Tensor:
    lr = float(optimizer.param_groups[0]["lr"])
    momentum = float(optimizer.param_groups[0].get("momentum", 0.0))
    if momentum <= 0.0:
        return lr * flat_grad

    buffers = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = optimizer.state.get(param, {})
            if "momentum_buffer" in state:
                buffers.append(state["momentum_buffer"].detach().reshape(-1))
            else:
                buffers.append(torch.zeros_like(param).reshape(-1))
    if len(buffers) == 0:
        return lr * flat_grad
    momentum_buffer = torch.cat(buffers, dim=0).to(flat_grad.device)
    if momentum_buffer.shape[0] != flat_grad.shape[0]:
        return lr * flat_grad
    return lr * (flat_grad + momentum * momentum_buffer)


def warmup_and_collect_affinity(config: CN, logger, working_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train, dataset_val, data_loader_train, data_loader_val, _ = build_loader(config)
    model = build_task_model(config, device, logger)
    criterion, loss_ft, _ = build_loss_bundle(config)
    optimizer = build_optimizer(config, model)

    warmup_epochs = int(config.MODEL.AGMTLORA.AFFINITY_COLLECT_EPOCHS)
    if warmup_epochs > 0:
        for _ in range(warmup_epochs):
            model.train()
            for batch in data_loader_train:
                samples, targets = move_batch_to_device(batch, config.TASKS, device)
                outputs = model(samples)
                loss, _ = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    warmup_checkpoint_path = os.path.join(working_dir, "warmup_checkpoint.pth")
    mkdir_if_missing(os.path.dirname(warmup_checkpoint_path))
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": max(0, warmup_epochs - 1),
            "extra_state": {
                "stage": "ag_mtlora_stage1_warmup",
                "tasks": list(config.TASKS),
            },
        },
        warmup_checkpoint_path,
    )

    shared_params = get_shared_ta_parameters(model)
    task_index = {task: idx for idx, task in enumerate(config.TASKS)}
    directed_sum = torch.zeros((len(config.TASKS), len(config.TASKS)), dtype=torch.float32, device=device)
    affinity_batches = 0

    model.train()
    for batch in data_loader_train:
        samples, targets = move_batch_to_device(batch, config.TASKS, device)
        optimizer.zero_grad()
        outputs = model(samples)

        flat_gradients = {}
        pseudo_updates = {}
        for task in config.TASKS:
            task_loss = loss_ft[task](outputs[task], targets[task])
            grads = torch.autograd.grad(
                task_loss,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            flat_grad = flatten_gradient_list(shared_params, grads)
            flat_gradients[task] = flat_grad
            pseudo_updates[task] = build_pseudo_update(flat_grad, optimizer)

        lr = float(optimizer.param_groups[0]["lr"])
        for src_task in config.TASKS:
            src_idx = task_index[src_task]
            for dst_task in config.TASKS:
                dst_idx = task_index[dst_task]
                directed_sum[src_idx, dst_idx] += torch.dot(
                    flat_gradients[dst_task], pseudo_updates[src_task]
                ) / max(lr, 1e-12)
        affinity_batches += 1

        loss, _ = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    directed_affinity = (directed_sum / max(float(affinity_batches), 1.0)).detach().cpu().numpy()
    symmetric_affinity = 0.5 * (directed_affinity + directed_affinity.T)
    val_losses = evaluate_task_losses(model, data_loader_val, loss_ft, config.TASKS, device)
    warmup_state_dict = clone_state_dict_to_cpu(model)

    del dataset_train, dataset_val, data_loader_train, data_loader_val
    del optimizer, criterion, loss_ft, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "warmup_checkpoint_path": warmup_checkpoint_path,
        "warmup_state_dict": warmup_state_dict,
        "directed_affinity": directed_affinity,
        "symmetric_affinity": symmetric_affinity,
        "validation_losses": val_losses,
        "num_batches": affinity_batches,
    }


def build_group_proxy(directed_affinity: np.ndarray, tasks: Sequence[str], candidate_groups: Sequence[Sequence[str]]):
    task_index = {task: idx for idx, task in enumerate(tasks)}
    group_proxy = {}
    csv_rows = []
    for group in candidate_groups:
        group = list(group)
        group_key = group_to_key(group)
        group_proxy[group_key] = {}
        for task in group:
            if len(group) == 1:
                proxy_value = 0.0
            else:
                incoming = [
                    float(directed_affinity[task_index[other_task], task_index[task]])
                    for other_task in group if other_task != task
                ]
                proxy_value = float(np.mean(incoming)) if len(incoming) > 0 else 0.0
            group_proxy[group_key][task] = proxy_value
            csv_rows.append({"group": group_key, "task": task, "proxy": proxy_value})
    return group_proxy, csv_rows


def collect_predictor_training_targets(
    base_config: CN,
    logger,
    warmup_state_dict: Dict[str, torch.Tensor],
    train_groups: Sequence[Sequence[str]],
    working_dir: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    singleton_losses = {}
    predictor_targets = {}

    singleton_groups = canonicalize_groups(
        [group for group in train_groups if len(group) == 1],
        base_config.TASKS,
    )
    for singleton_group in singleton_groups:
        task = singleton_group[0]
        task_output_dir = os.path.join(working_dir, "predictor_runs", f"singleton__{task}")
        task_config = rebuild_task_config(base_config, [task], output_dir=task_output_dir)
        _, val_losses = short_train_model(
            task_config,
            logger,
            warmup_state_dict,
            task_config.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS,
            device,
        )
        singleton_losses[task] = float(val_losses[task])
        predictor_targets[group_to_key([task])] = {task: 0.0}

    for group in train_groups:
        group = list(group)
        if len(group) == 1:
            continue
        group_key = group_to_key(group)
        group_output_dir = os.path.join(working_dir, "predictor_runs", f"group__{group_key.replace('|', '__')}")
        group_config = rebuild_task_config(base_config, group, output_dir=group_output_dir)
        _, val_losses = short_train_model(
            group_config,
            logger,
            warmup_state_dict,
            group_config.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS,
            device,
        )
        predictor_targets[group_key] = {}
        for task in group:
            predictor_targets[group_key][task] = float(singleton_losses[task] - val_losses[task])

    return predictor_targets, singleton_losses


def fit_base_predictor(train_pairs: List[Dict], predictor_name: str, predictor_kwargs: Dict):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer
    except ImportError as exc:
        raise ImportError("AG-MTLoRA Stage-1 requires scikit-learn for the predictor chain.") from exc

    x = np.array([row["proxy"] for row in train_pairs], dtype=np.float64).reshape(-1, 1)
    y = np.array([row["gain"] for row in train_pairs], dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError("No training pairs were collected for the base predictor.")

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x.shape[0] >= 2 and x_std > 1e-12:
        correlation = float(np.corrcoef(x[:, 0], y)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    if x_std <= 1e-12:
        affine_a = 1.0
    else:
        affine_a = correlation * (y_std / max(x_std, 1e-12))
    affine_b = y_mean - affine_a * x_mean
    calibrated_x = affine_a * x + affine_b

    predictor_name = str(predictor_name)
    predictor_kwargs = dict(predictor_kwargs or {})
    if predictor_name == "spline_ridge":
        model = Pipeline(
            steps=[
                (
                    "spline",
                    SplineTransformer(
                        degree=int(predictor_kwargs.get("degree", 3)),
                        n_knots=int(predictor_kwargs.get("n_knots", 4)),
                        include_bias=False,
                    ),
                ),
                ("ridge", Ridge(alpha=float(predictor_kwargs.get("alpha", 1.0)))),
            ]
        )
    elif predictor_name == "knn":
        model = KNeighborsRegressor(n_neighbors=int(predictor_kwargs.get("n_neighbors", 3)))
    elif predictor_name == "rf":
        model = RandomForestRegressor(
            n_estimators=int(predictor_kwargs.get("n_estimators", 200)),
            random_state=int(predictor_kwargs.get("random_state", 0)),
        )
    else:
        raise ValueError(f"Unsupported BASE_PREDICTOR: {predictor_name}")

    model.fit(calibrated_x, y)
    return {
        "predictor_name": predictor_name,
        "affine_a": affine_a,
        "affine_b": affine_b,
        "model": model,
    }


def predict_base_gain(base_predictor, proxy_value: float) -> float:
    calibrated_x = np.array([[base_predictor["affine_a"] * proxy_value + base_predictor["affine_b"]]], dtype=np.float64)
    prediction = base_predictor["model"].predict(calibrated_x)
    return float(np.asarray(prediction).reshape(-1)[0])


def fit_residual_predictor(tasks: Sequence[str], train_groups: Sequence[Sequence[str]], predictor_targets: Dict[str, Dict[str, float]], initial_predictions: Dict[str, Dict[str, float]], alpha: float):
    try:
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise ImportError("AG-MTLoRA Stage-1 requires scikit-learn for the residual predictor.") from exc

    x_rows = []
    y_rows = []
    for group in train_groups:
        group = list(group)
        group_key = group_to_key(group)
        mask = np.array([1.0 if task in group else 0.0 for task in tasks], dtype=np.float64)
        residual = np.zeros(len(tasks), dtype=np.float64)
        for task_idx, task in enumerate(tasks):
            if task in predictor_targets[group_key]:
                residual[task_idx] = predictor_targets[group_key][task] - initial_predictions[group_key].get(task, 0.0)
        x_rows.append(mask)
        y_rows.append(residual)

    model = Ridge(alpha=float(alpha))
    model.fit(np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64))
    return model


def predict_all_candidate_groups(
    tasks: Sequence[str],
    candidate_groups: Sequence[Sequence[str]],
    group_proxy: Dict[str, Dict[str, float]],
    base_predictor,
    residual_model,
):
    initial_predictions = {}
    residual_predictions = {}
    final_predictions = {}

    for group in candidate_groups:
        group = list(group)
        group_key = group_to_key(group)
        initial_predictions[group_key] = {task: 0.0 for task in group}
        if len(group) == 1:
            task = group[0]
            initial_predictions[group_key][task] = 0.0
            residual_predictions[group_key] = {task: 0.0 for task in group}
            final_predictions[group_key] = {task: 0.0 for task in group}
            continue

        for task in group:
            initial_predictions[group_key][task] = predict_base_gain(
                base_predictor,
                group_proxy[group_key][task],
            )

        mask = np.array([[1.0 if task in group else 0.0 for task in tasks]], dtype=np.float64)
        residual_vector = residual_model.predict(mask)[0]
        residual_predictions[group_key] = {
            task: float(residual_vector[tasks.index(task)]) for task in group
        }
        final_predictions[group_key] = {
            task: float(initial_predictions[group_key][task] + residual_predictions[group_key][task])
            for task in group
        }

    return initial_predictions, residual_predictions, final_predictions


def run_partition_search(tasks: Sequence[str], final_predictions: Dict[str, Dict[str, float]], max_groups: int):
    partitions = enumerate_partitions(tasks, max_groups)
    ranked_results = []
    for partition in partitions:
        partition = canonicalize_groups(partition, tasks)
        per_task_scores = {}
        for group in partition:
            group_key = group_to_key(group)
            for task in group:
                per_task_scores[task] = float(final_predictions[group_key][task])
        score = float(np.mean([per_task_scores[task] for task in tasks]))
        ranked_results.append(
            {
                "groups": partition,
                "partition_score": score,
                "per_task_scores": per_task_scores,
                "num_groups": len(partition),
            }
        )

    ranked_results.sort(
        key=lambda item: (
            -item["partition_score"],
            item["num_groups"],
            [group_to_key(group) for group in item["groups"]],
        )
    )
    return ranked_results


def create_resolved_training_config(base_config: CN, grouping_json_path: str, resolved_group_ranks: List[List[int]]) -> CN:
    resolved_config = base_config.clone()
    resolved_config.defrost()
    resolved_config.MODEL.AGMTLORA.ENABLED = True
    resolved_config.MODEL.AGMTLORA.STAGE = 1
    resolved_config.MODEL.AGMTLORA.GROUPING_SOURCE = "fixed_json"
    resolved_config.MODEL.AGMTLORA.GROUPING_JSON = os.path.abspath(grouping_json_path)
    resolved_config.MODEL.AGMTLORA.GROUP_SHARED_RANKS = resolved_group_ranks
    resolved_config.freeze()
    return resolved_config


def run_stage1_pipeline(config: CN, output_root: str, logger):
    mkdir_if_missing(output_root)

    affinity_result = warmup_and_collect_affinity(config, logger, output_root)
    directed_affinity = affinity_result["directed_affinity"]
    symmetric_affinity = affinity_result["symmetric_affinity"]

    affinity_json_path = config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH
    affinity_csv_path = os.path.splitext(affinity_json_path)[0] + ".csv"
    affinity_sym_json_path = os.path.join(os.path.dirname(affinity_json_path), "affinity_symmetric.json")
    affinity_sym_csv_path = os.path.join(os.path.dirname(affinity_json_path), "affinity_symmetric.csv")

    save_json(
        {
            "tasks": list(config.TASKS),
            "directed_affinity": directed_affinity.tolist(),
            "num_batches": int(affinity_result["num_batches"]),
            "warmup_checkpoint": affinity_result["warmup_checkpoint_path"],
            "warmup_validation_losses": affinity_result["validation_losses"],
        },
        affinity_json_path,
    )
    save_matrix_csv(config.TASKS, directed_affinity, affinity_csv_path)

    if config.MODEL.AGMTLORA.VISUALIZE_SYMMETRIC_AFFINITY:
        save_json(
            {
                "tasks": list(config.TASKS),
                "symmetric_affinity": symmetric_affinity.tolist(),
            },
            affinity_sym_json_path,
        )
        save_matrix_csv(config.TASKS, symmetric_affinity, affinity_sym_csv_path)

    candidate_groups = enumerate_candidate_groups(config.TASKS)
    group_proxy, group_proxy_rows = build_group_proxy(directed_affinity, config.TASKS, candidate_groups)
    group_proxy_json_path = os.path.join(output_root, "group_proxy.json")
    group_proxy_csv_path = os.path.join(output_root, "group_proxy.csv")
    save_json({"tasks": list(config.TASKS), "group_proxy": group_proxy}, group_proxy_json_path)
    save_group_task_rows(group_proxy_rows, group_proxy_csv_path, "proxy")

    predictor_train_groups = select_predictor_train_groups(
        config.TASKS,
        int(config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_BUDGET),
        config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_STRATEGY,
        int(config.SEED),
    )
    predictor_targets, singleton_losses = collect_predictor_training_targets(
        config,
        logger,
        affinity_result["warmup_state_dict"],
        predictor_train_groups,
        output_root,
    )
    predictor_train_groups_json = os.path.join(output_root, "predictor_train_groups.json")
    predictor_train_groups_csv = os.path.join(output_root, "predictor_train_groups.csv")
    save_json(
        {
            "tasks": list(config.TASKS),
            "train_groups": predictor_train_groups,
            "singleton_val_losses": singleton_losses,
            "gt_predicted_gains": predictor_targets,
        },
        predictor_train_groups_json,
    )
    save_predictor_value_csv(predictor_targets, predictor_train_groups_csv)

    base_train_pairs = []
    for group in predictor_train_groups:
        group_key = group_to_key(group)
        for task in group:
            base_train_pairs.append(
                {
                    "group": group_key,
                    "task": task,
                    "proxy": float(group_proxy[group_key][task]),
                    "gain": float(predictor_targets[group_key][task]),
                }
            )

    base_predictor = fit_base_predictor(
        base_train_pairs,
        config.MODEL.AGMTLORA.BASE_PREDICTOR,
        dict(config.MODEL.AGMTLORA.BASE_PREDICTOR_KWARGS),
    )

    initial_train_predictions = {}
    for group in predictor_train_groups:
        group_key = group_to_key(group)
        initial_train_predictions[group_key] = {}
        for task in group:
            initial_train_predictions[group_key][task] = predict_base_gain(
                base_predictor,
                group_proxy[group_key][task],
            )

    residual_model = fit_residual_predictor(
        config.TASKS,
        predictor_train_groups,
        predictor_targets,
        initial_train_predictions,
        float(config.MODEL.AGMTLORA.RESIDUAL_ALPHA),
    )

    initial_predictions, residual_predictions, final_predictions = predict_all_candidate_groups(
        config.TASKS,
        candidate_groups,
        group_proxy,
        base_predictor,
        residual_model,
    )

    initial_predictions_json = os.path.join(output_root, "initial_predictions.json")
    residual_predictions_json = os.path.join(output_root, "residual_predictions.json")
    final_predictions_json = os.path.join(output_root, "final_predictions.json")
    save_json({"tasks": list(config.TASKS), "initial_predictions": initial_predictions}, initial_predictions_json)
    save_json({"tasks": list(config.TASKS), "residual_predictions": residual_predictions}, residual_predictions_json)
    save_json({"tasks": list(config.TASKS), "final_predictions": final_predictions}, final_predictions_json)
    save_predictor_value_csv(initial_predictions, os.path.join(output_root, "initial_predictions.csv"))
    save_predictor_value_csv(residual_predictions, os.path.join(output_root, "residual_predictions.csv"))
    save_predictor_value_csv(final_predictions, os.path.join(output_root, "final_predictions.csv"))

    ranked_partitions = run_partition_search(
        config.TASKS,
        final_predictions,
        int(config.MODEL.AGMTLORA.MAX_GROUPS),
    )
    partition_results_path = os.path.join(output_root, "partition_search_results.json")
    partition_results_csv = os.path.join(output_root, "partition_search_results.csv")
    save_json(
        {
            "tasks": list(config.TASKS),
            "search_objective": str(config.MODEL.AGMTLORA.SEARCH_OBJECTIVE),
            "ranked_partitions": ranked_partitions,
        },
        partition_results_path,
    )
    save_partition_csv(ranked_partitions, partition_results_csv)

    best_partition = ranked_partitions[0]
    resolved_group_ranks, rank_source = resolve_group_shared_ranks(
        config.MODEL.AGMTLORA.GROUP_SHARED_RANKS,
        int(config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET),
        len(best_partition["groups"]),
        len(config.MODEL.SWIN.DEPTHS),
        str(config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION),
    )
    task_to_group = build_task_to_group(best_partition["groups"])
    grouping_payload = {
        "tasks": list(config.TASKS),
        "groups": best_partition["groups"],
        "task_to_group": task_to_group,
        "max_groups": int(config.MODEL.AGMTLORA.MAX_GROUPS),
        "partition_score": float(best_partition["partition_score"]),
        "group_shared_ranks": resolved_group_ranks,
        "search_objective": str(config.MODEL.AGMTLORA.SEARCH_OBJECTIVE),
        "affinity_path": affinity_json_path,
        "final_predictions_path": final_predictions_json,
        "warmup_checkpoint": affinity_result["warmup_checkpoint_path"],
        "partition_search_results": partition_results_path,
        "group_rank_source": rank_source,
    }
    grouping_json_path = config.MODEL.AGMTLORA.GROUPING_SAVE_PATH
    save_json(grouping_payload, grouping_json_path)

    resolved_config = create_resolved_training_config(config, grouping_json_path, resolved_group_ranks)
    resolved_config_path = os.path.join(output_root, "resolved_agmtlora_config.yaml")
    mkdir_if_missing(os.path.dirname(resolved_config_path))
    with open(resolved_config_path, "w", encoding="utf-8") as handle:
        handle.write(resolved_config.dump())

    return {
        "affinity_json_path": affinity_json_path,
        "group_proxy_json_path": group_proxy_json_path,
        "predictor_train_groups_json": predictor_train_groups_json,
        "predictor_train_groups_csv": predictor_train_groups_csv,
        "initial_predictions_json": initial_predictions_json,
        "residual_predictions_json": residual_predictions_json,
        "final_predictions_json": final_predictions_json,
        "partition_results_path": partition_results_path,
        "partition_results_csv": partition_results_csv,
        "grouping_json_path": grouping_json_path,
        "resolved_config_path": resolved_config_path,
        "warmup_checkpoint_path": affinity_result["warmup_checkpoint_path"],
    }


def create_stage1_logger(output_root: str):
    mkdir_if_missing(output_root)
    return create_logger(output_dir=output_root, dist_rank=0, name="ag_mtlora_stage1")
