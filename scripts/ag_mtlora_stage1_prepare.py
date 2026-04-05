import argparse
import datetime
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ag_mtlora.stage1 import create_stage1_logger, run_stage1_pipeline, set_random_seed
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser("AG-MTLoRA Stage-1 preparation")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--resume-backbone", type=str)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default="output", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fused_window_process", action="store_true")
    parser.add_argument("--fused_layernorm", action="store_true")
    parser.add_argument("--optim", type=str)
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task list.")
    parser.add_argument("--nyud", type=str)
    parser.add_argument("--pascal", type=str)
    parser.add_argument("--decoder_map", type=str)
    parser.add_argument("--skip_decoder", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(config.OUTPUT, "ag_mtlora_stage1_prepare", f"run_{timestamp}")
    config.defrost()
    config.OUTPUT = output_root
    config.MODEL.AGMTLORA.AFFINITY_SAVE_PATH = os.path.join(output_root, "affinity.json")
    config.MODEL.AGMTLORA.GROUPING_SAVE_PATH = os.path.join(output_root, "grouping.json")
    config.freeze()
    logger = create_stage1_logger(output_root)

    set_random_seed(int(config.SEED))
    logger.info("Running AG-MTLoRA Stage-1 preparation with config:\n%s", config.dump())
    logger.info("CLI args: %s", json.dumps(vars(args), ensure_ascii=False))

    artifacts = run_stage1_pipeline(config, output_root, logger)
    logger.info("AG-MTLoRA Stage-1 preparation finished.")
    logger.info(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
