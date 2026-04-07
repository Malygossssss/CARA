import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml
from yacs.config import CfgNode as CN

import ag_mtlora.stage1 as stage1


class FakeDataset:
    def __init__(self, sample_ids):
        self.im_ids = list(sample_ids)

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, index):
        return {"image": index}


class DummyResolvedConfig:
    def dump(self):
        return "MODEL:\n  AGMTLORA:\n    ENABLED: true\n"


def build_stage1_config(tmpdir, split_mode="train_meta_strict"):
    config = CN()
    config.SEED = 7
    config.TASKS = ["task_a", "task_b"]

    config.DATA = CN()
    config.DATA.DBNAME = "NYUD"
    config.DATA.DATA_PATH = tmpdir
    config.DATA.BATCH_SIZE = 2
    config.DATA.NUM_WORKERS = 0
    config.DATA.PIN_MEMORY = False

    config.MODEL = CN()
    config.MODEL.SWIN = CN()
    config.MODEL.SWIN.DEPTHS = [2]
    config.MODEL.AGMTLORA = CN(new_allowed=True)
    config.MODEL.AGMTLORA.DATA_SPLIT_MODE = split_mode
    config.MODEL.AGMTLORA.META_VAL_RATIO = 0.2
    config.MODEL.AGMTLORA.RESOLVED_META_SPLIT_SEED = 13
    config.MODEL.AGMTLORA.META_SPLIT_SAVE_PATH = os.path.join(tmpdir, "meta_split.json")
    config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = os.path.join(tmpdir, "affinity.json")
    config.MODEL.AGMTLORA.GROUPING_SAVE_PATH = os.path.join(tmpdir, "grouping.json")
    config.MODEL.AGMTLORA.VISUALIZE_SYMMETRIC_AFFINITY = False
    config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_BUDGET = 3
    config.MODEL.AGMTLORA.PREDICTOR_TRAIN_GROUP_STRATEGY = "all_singletons+all_pairs+random_higher_order"
    config.MODEL.AGMTLORA.BASE_PREDICTOR = "spline_ridge"
    config.MODEL.AGMTLORA.BASE_PREDICTOR_KWARGS = CN(new_allowed=True)
    config.MODEL.AGMTLORA.RESIDUAL_ALPHA = 1.0
    config.MODEL.AGMTLORA.MAX_GROUPS = 2
    config.MODEL.AGMTLORA.TOTAL_SHARED_RANK_BUDGET = 2
    config.MODEL.AGMTLORA.GROUP_SHARED_RANKS = []
    config.MODEL.AGMTLORA.GROUP_RANK_ALLOCATION = "equal_split"
    config.MODEL.AGMTLORA.SEARCH_OBJECTIVE = "mean_final_predicted_gain"

    config.TASKS_CONFIG = CN(new_allowed=True)
    config.TASKS_CONFIG.ALL_TASKS = CN(new_allowed=True)
    config.TASKS_CONFIG.ALL_TASKS.FLAGVALS = CN(new_allowed=True)
    config.TASKS_CONFIG.FLAGVALS = CN(new_allowed=True)

    config.OUTPUT = tmpdir
    config.freeze()
    return config


class Stage1MetaSplitTest(unittest.TestCase):
    def test_parse_predictor_progress_from_log_recovers_singletons_and_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "log_rank0.txt")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("Predictor target collected | singleton task=semseg | val_loss=1.234500\n")
                handle.write("Predictor target collected | group=['semseg', 'normals'] | gains={'semseg': 0.1, 'normals': 0.2}\n")

            predictor_targets, singleton_losses = stage1.parse_predictor_progress_from_log(log_path)

            self.assertEqual(singleton_losses, {"semseg": 1.2345})
            self.assertEqual(predictor_targets["semseg"], {"semseg": 0.0})
            self.assertEqual(predictor_targets["semseg|normals"], {"semseg": 0.1, "normals": 0.2})

    def test_build_stage1_data_split_manifest_is_deterministic_and_persists_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            sample_ids = [f"sample_{idx}" for idx in range(10)]

            with mock.patch(
                "ag_mtlora.stage1.get_mtl_split_sample_ids",
                return_value=sample_ids,
            ):
                logger = mock.Mock()
                manifest_a = stage1.build_stage1_data_split_manifest(config, logger)
                manifest_b = stage1.build_stage1_data_split_manifest(config, logger)

            self.assertEqual(manifest_a["meta_train_indices"], manifest_b["meta_train_indices"])
            self.assertEqual(manifest_a["meta_val_indices"], manifest_b["meta_val_indices"])
            self.assertEqual(len(manifest_a["meta_val_indices"]), 2)
            self.assertEqual(len(manifest_a["meta_train_indices"]), 8)
            self.assertEqual(
                sorted(manifest_a["meta_train_indices"] + manifest_a["meta_val_indices"]),
                list(range(10)),
            )
            self.assertTrue(set(manifest_a["meta_train_indices"]).isdisjoint(manifest_a["meta_val_indices"]))
            self.assertTrue(os.path.exists(manifest_a["meta_split_path"]))

            with open(manifest_a["meta_split_path"], "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["meta_train_ids"], manifest_a["meta_train_ids"])
            self.assertEqual(persisted["meta_val_ids"], manifest_a["meta_val_ids"])

    def test_build_stage1_data_loaders_strict_uses_train_split_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            sample_ids = [f"sample_{idx}" for idx in range(6)]
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "meta_train_indices": [0, 2, 4, 5],
                "meta_val_indices": [1, 3],
                "meta_train_ids": [sample_ids[idx] for idx in [0, 2, 4, 5]],
                "meta_val_ids": [sample_ids[idx] for idx in [1, 3]],
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }

            with mock.patch("ag_mtlora.stage1.get_transformations", return_value=("train_tf", "eval_tf")), mock.patch(
                "ag_mtlora.stage1.get_mtl_dataset",
                side_effect=lambda db_name, cfg, transforms, split: FakeDataset(sample_ids),
            ) as mocked_get_dataset, mock.patch(
                "ag_mtlora.stage1.get_mtl_train_dataloader",
                side_effect=lambda cfg, dataset: ("train_loader", len(dataset)),
            ) as mocked_train_loader, mock.patch(
                "ag_mtlora.stage1.get_mtl_val_dataloader",
                side_effect=lambda cfg, dataset: ("val_loader", len(dataset)),
            ) as mocked_val_loader:
                dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = stage1.build_stage1_data_loaders(
                    config,
                    data_split_manifest=manifest,
                )

            self.assertEqual(len(dataset_train), 4)
            self.assertEqual(len(dataset_val), 2)
            self.assertEqual(data_loader_train, ("train_loader", 4))
            self.assertEqual(data_loader_val, ("val_loader", 2))
            self.assertIsNone(mixup_fn)
            self.assertEqual(mocked_get_dataset.call_count, 2)
            self.assertTrue(all(call.kwargs["split"] == "train" for call in mocked_get_dataset.call_args_list))
            mocked_train_loader.assert_called_once()
            mocked_val_loader.assert_called_once()

    def test_build_stage1_data_loaders_projects_manifest_ids_for_filtered_task_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "meta_train_indices": [0, 2, 4, 5],
                "meta_val_indices": [1, 3],
                "meta_train_ids": ["sample_0", "sample_2", "sample_4", "sample_5"],
                "meta_val_ids": ["sample_1", "sample_3"],
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }
            filtered_sample_ids = ["sample_1", "sample_2", "sample_5"]

            with mock.patch("ag_mtlora.stage1.get_transformations", return_value=("train_tf", "eval_tf")), mock.patch(
                "ag_mtlora.stage1.get_mtl_dataset",
                side_effect=lambda db_name, cfg, transforms, split: FakeDataset(filtered_sample_ids),
            ), mock.patch(
                "ag_mtlora.stage1.get_mtl_train_dataloader",
                side_effect=lambda cfg, dataset: ("train_loader", list(dataset.indices)),
            ), mock.patch(
                "ag_mtlora.stage1.get_mtl_val_dataloader",
                side_effect=lambda cfg, dataset: ("val_loader", list(dataset.indices)),
            ):
                dataset_train, dataset_val, data_loader_train, data_loader_val, _ = stage1.build_stage1_data_loaders(
                    config,
                    data_split_manifest=manifest,
                )

            self.assertEqual(list(dataset_train.indices), [1, 2])
            self.assertEqual(list(dataset_val.indices), [0])
            self.assertEqual(data_loader_train, ("train_loader", [1, 2]))
            self.assertEqual(data_loader_val, ("val_loader", [0]))

    def test_create_resolved_training_config_is_schema_safe_and_base_backed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg_path = os.path.join(tmpdir, "base.yaml")
            grouping_json_path = os.path.join(tmpdir, "grouping.json")
            with open(base_cfg_path, "w", encoding="utf-8") as handle:
                handle.write("MODEL:\n  NAME: test\n")

            resolved = stage1.create_resolved_training_config(
                base_cfg_path,
                grouping_json_path,
                [[32, 32], [16, 16]],
            )

            payload = yaml.safe_load(resolved.dump())
            self.assertEqual(payload["BASE"], [os.path.abspath(base_cfg_path)])
            self.assertEqual(payload["MODEL"]["AGMTLORA"]["GROUPING_SOURCE"], "fixed_json")
            self.assertEqual(payload["MODEL"]["AGMTLORA"]["GROUPING_JSON"], os.path.abspath(grouping_json_path))
            self.assertEqual(payload["MODEL"]["AGMTLORA"]["GROUP_SHARED_RANKS"], [[32, 32], [16, 16]])

    def test_build_stage1_data_loaders_legacy_mode_uses_existing_build_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir, split_mode="official_val")
            expected = ("dataset_train", "dataset_val", "loader_train", "loader_val", None)

            with mock.patch("ag_mtlora.stage1.build_loader", return_value=expected) as mocked_build_loader:
                result = stage1.build_stage1_data_loaders(
                    config,
                    data_split_manifest={"selection_data_split_mode": "official_val"},
                )

            self.assertEqual(result, expected)
            mocked_build_loader.assert_called_once_with(config)

    def test_collect_predictor_training_targets_reuses_same_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = CN()
            base_config.TASKS = ["task_a", "task_b"]
            manifest = {"selection_data_split_mode": "train_meta_strict"}

            def make_group_config(group):
                cfg = CN()
                cfg.TASKS = list(group)
                cfg.MODEL = CN()
                cfg.MODEL.AGMTLORA = CN()
                cfg.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS = 1
                cfg.OUTPUT = tmpdir
                return cfg

            short_train_side_effects = [
                (None, {"task_a": 1.0}),
                (None, {"task_b": 2.0}),
                (None, {"task_a": 0.7, "task_b": 1.6}),
            ]

            with mock.patch(
                "ag_mtlora.stage1.rebuild_task_config",
                side_effect=lambda config, group, output_dir=None: make_group_config(group),
            ), mock.patch(
                "ag_mtlora.stage1.short_train_model",
                side_effect=short_train_side_effects,
            ) as mocked_short_train:
                predictor_targets, singleton_losses = stage1.collect_predictor_training_targets(
                    base_config,
                    mock.Mock(),
                    baseline_state_dict={},
                    train_groups=[["task_a"], ["task_b"], ["task_a", "task_b"]],
                    working_dir=tmpdir,
                    data_split_manifest=manifest,
                )

            self.assertEqual(singleton_losses, {"task_a": 1.0, "task_b": 2.0})
            self.assertEqual(predictor_targets["task_a"], {"task_a": 0.0})
            self.assertEqual(predictor_targets["task_b"], {"task_b": 0.0})
            self.assertAlmostEqual(predictor_targets["task_a|task_b"]["task_a"], 0.3)
            self.assertAlmostEqual(predictor_targets["task_a|task_b"]["task_b"], 0.4)
            self.assertEqual(mocked_short_train.call_count, 3)
            self.assertTrue(all(call.kwargs["data_split_manifest"] is manifest for call in mocked_short_train.call_args_list))

    def test_collect_predictor_training_targets_uses_cached_log_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "log_rank0.txt")
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("Predictor target collected | singleton task=task_a | val_loss=1.000000\n")

            base_config = CN()
            base_config.TASKS = ["task_a", "task_b"]
            manifest = {"selection_data_split_mode": "train_meta_strict", "meta_split_path": os.path.join(tmpdir, "meta_split.json")}

            def make_group_config(group):
                cfg = CN()
                cfg.TASKS = list(group)
                cfg.MODEL = CN()
                cfg.MODEL.AGMTLORA = CN()
                cfg.MODEL.AGMTLORA.PREDICTOR_GROUP_TRAIN_EPOCHS = 1
                cfg.OUTPUT = tmpdir
                return cfg

            with mock.patch(
                "ag_mtlora.stage1.rebuild_task_config",
                side_effect=lambda config, group, output_dir=None: make_group_config(group),
            ), mock.patch(
                "ag_mtlora.stage1.short_train_model",
                side_effect=[
                    (None, {"task_b": 2.0}),
                    (None, {"task_a": 0.8, "task_b": 1.5}),
                ],
            ) as mocked_short_train:
                predictor_targets, singleton_losses = stage1.collect_predictor_training_targets(
                    base_config,
                    mock.Mock(),
                    baseline_state_dict={},
                    train_groups=[["task_a"], ["task_b"], ["task_a", "task_b"]],
                    working_dir=tmpdir,
                    data_split_manifest=manifest,
                )

            self.assertEqual(singleton_losses["task_a"], 1.0)
            self.assertEqual(singleton_losses["task_b"], 2.0)
            self.assertEqual(mocked_short_train.call_count, 2)
            self.assertEqual(predictor_targets["task_a"], {"task_a": 0.0})
            self.assertEqual(predictor_targets["task_b"], {"task_b": 0.0})

    def test_run_stage1_pipeline_records_split_metadata_and_returns_meta_split_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_stage1_config(tmpdir)
            logger = mock.Mock()
            manifest = {
                "selection_data_split_mode": "train_meta_strict",
                "selection_eval_label": "meta-val",
                "meta_split_path": os.path.join(tmpdir, "meta_split.json"),
            }
            directed_affinity = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
            final_predictions = {
                "task_a": {"task_a": 0.0},
                "task_b": {"task_b": 0.0},
                "task_a|task_b": {"task_a": 0.4, "task_b": 0.6},
            }

            with mock.patch(
                "ag_mtlora.stage1.build_stage1_data_split_manifest",
                return_value=manifest,
            ) as mocked_manifest, mock.patch(
                "ag_mtlora.stage1.warmup_and_collect_affinity",
                return_value={
                    "directed_affinity": directed_affinity,
                    "symmetric_affinity": 0.5 * (directed_affinity + directed_affinity.T),
                    "num_affinity_epochs": 1,
                    "num_batches_per_epoch": [3],
                    "affinity_epoch_history": [directed_affinity],
                    "warmup_checkpoint_path": os.path.join(tmpdir, "warmup_checkpoint.pth"),
                    "post_affinity_checkpoint_path": os.path.join(tmpdir, "post_affinity_checkpoint.pth"),
                    "warmup_validation_losses": {"task_a": 1.0, "task_b": 2.0},
                    "post_affinity_validation_losses": {"task_a": 0.8, "task_b": 1.7},
                    "post_affinity_state_dict": {},
                },
            ) as mocked_warmup, mock.patch(
                "ag_mtlora.stage1.enumerate_candidate_groups",
                return_value=[["task_a"], ["task_b"], ["task_a", "task_b"]],
            ), mock.patch(
                "ag_mtlora.stage1.build_group_proxy",
                return_value=(
                    {
                        "task_a": {"task_a": 0.0},
                        "task_b": {"task_b": 0.0},
                        "task_a|task_b": {"task_a": 1.0, "task_b": 2.0},
                    },
                    [],
                ),
            ), mock.patch(
                "ag_mtlora.stage1.select_predictor_train_groups",
                return_value=[["task_a"], ["task_b"], ["task_a", "task_b"]],
            ), mock.patch(
                "ag_mtlora.stage1.collect_predictor_training_targets",
                return_value=(
                    {
                        "task_a": {"task_a": 0.0},
                        "task_b": {"task_b": 0.0},
                        "task_a|task_b": {"task_a": 0.2, "task_b": 0.3},
                    },
                    {"task_a": 1.0, "task_b": 1.5},
                ),
            ) as mocked_collect, mock.patch(
                "ag_mtlora.stage1.fit_base_predictor",
                return_value={
                    "predictor_name": "mock",
                    "num_samples": 4,
                    "affine_a": 1.0,
                    "affine_b": 0.0,
                },
            ), mock.patch(
                "ag_mtlora.stage1.predict_base_gain",
                side_effect=lambda predictor, proxy_value: float(proxy_value),
            ), mock.patch(
                "ag_mtlora.stage1.fit_residual_predictor",
                return_value=object(),
            ), mock.patch(
                "ag_mtlora.stage1.predict_all_candidate_groups",
                return_value=(
                    final_predictions,
                    final_predictions,
                    final_predictions,
                ),
            ), mock.patch(
                "ag_mtlora.stage1.run_partition_search",
                return_value=[
                    {
                        "groups": [["task_a", "task_b"]],
                        "partition_score": 0.5,
                        "per_task_scores": {"task_a": 0.4, "task_b": 0.6},
                        "num_groups": 1,
                    }
                ],
            ), mock.patch(
                "ag_mtlora.stage1.resolve_group_shared_ranks",
                return_value=([[2]], "auto_equal_split"),
            ), mock.patch(
                "ag_mtlora.stage1.build_task_to_group",
                return_value={"task_a": "group_0", "task_b": "group_0"},
            ), mock.patch(
                "ag_mtlora.stage1.create_resolved_training_config",
                return_value=DummyResolvedConfig(),
            ), mock.patch(
                "ag_mtlora.stage1.save_matrix_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_affinity_epoch_history_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_group_task_rows"
            ), mock.patch(
                "ag_mtlora.stage1.save_predictor_value_csv"
            ), mock.patch(
                "ag_mtlora.stage1.save_partition_csv"
            ):
                artifacts = stage1.run_stage1_pipeline(
                    config,
                    tmpdir,
                    logger,
                    base_cfg_path=os.path.join(tmpdir, "base_config.yaml"),
                )

            self.assertEqual(artifacts["meta_split_path"], manifest["meta_split_path"])
            self.assertTrue(os.path.exists(artifacts["resolved_runtime_snapshot_path"]))
            mocked_manifest.assert_called_once_with(config, logger)
            mocked_warmup.assert_called_once()
            self.assertIs(mocked_warmup.call_args.kwargs["data_split_manifest"], manifest)
            self.assertIs(mocked_collect.call_args.kwargs["data_split_manifest"], manifest)

            with open(config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH, "r", encoding="utf-8") as handle:
                affinity_payload = json.load(handle)
            with open(os.path.join(tmpdir, "predictor_train_groups.json"), "r", encoding="utf-8") as handle:
                predictor_payload = json.load(handle)
            with open(config.MODEL.AGMTLORA.GROUPING_SAVE_PATH, "r", encoding="utf-8") as handle:
                grouping_payload = json.load(handle)

            for payload in (affinity_payload, predictor_payload, grouping_payload):
                self.assertEqual(payload["selection_data_split_mode"], "train_meta_strict")
                self.assertEqual(payload["meta_split_path"], manifest["meta_split_path"])


if __name__ == "__main__":
    unittest.main()
