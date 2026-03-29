import argparse
import json
import os

import torch

from config import get_config
from data import build_mtl_eval_loader
from logger import create_logger
from pruning.experiment import (
    build_model_for_experiment,
    evaluate_model,
    load_model_state,
    save_json,
    set_random_seed,
)


def parse_option():
    parser = argparse.ArgumentParser("UniPoRA pruned-checkpoint evaluation", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--checkpoint", "--resume", dest="checkpoint", required=True, help="checkpoint to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="evaluation split")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--disable_amp", action="store_true", help="disable pytorch amp")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic mode")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="root output folder")
    parser.add_argument("--output-dir", type=str, help="explicit directory to save eval artifacts")
    parser.add_argument("--name", type=str, help="override model name")
    parser.add_argument("--tag", help="experiment tag")
    parser.add_argument("--tasks", type=str, default="depth", help="comma separated tasks")
    parser.add_argument("--nyud", type=str, help="NYUD dataset path")
    parser.add_argument("--pascal", type=str, help="PASCAL dataset path")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    args = parser.parse_args()
    args.resume = args.checkpoint
    config = get_config(args)
    return args, config


def get_output_dir(args):
    if args.output_dir:
        return args.output_dir
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    return os.path.join(checkpoint_dir, f"standalone_eval_{args.split}")


if __name__ == "__main__":
    args, config = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(config.SEED, config.DETERMINISTIC)

    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, dist_rank=0, name="pruned_eval")
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as handle:
        handle.write(config.dump())

    logger.info("Running standalone pruned-checkpoint evaluation on %s", args.split)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Output dir: %s", output_dir)

    _, eval_loader = build_mtl_eval_loader(config, split=args.split)
    model = build_model_for_experiment(config, device)
    load_model_state(model, args.checkpoint, config, logger)
    summary = evaluate_model(config, eval_loader, model, device, logger, output_dir, args.split)

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "summary": summary,
    }
    save_json(os.path.join(output_dir, f"eval_{args.split}.json"), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
