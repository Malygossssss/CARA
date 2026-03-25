import argparse
import json

from config import get_config
from pruning import run_pruning_experiment


def parse_option():
    parser = argparse.ArgumentParser("UniPoRA TA-Prompt pruning pipeline", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", default=None, nargs="+", help="Modify config options by adding KEY VALUE pairs.")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--ckpt-freq", type=int, default=1, help="checkpoint saving frequency")
    parser.add_argument("--eval-freq", type=int, default=1, help="evaluation frequency")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--data-path", type=str, help="dataset path")
    parser.add_argument("--pretrained", help="pretrained weights")
    parser.add_argument("--resume", help="checkpoint to resume / use as teacher")
    parser.add_argument("--resume-backbone", help="resume checkpoint into the backbone")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true", help="use gradient checkpointing")
    parser.add_argument("--disable_amp", action="store_true", help="disable pytorch amp")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--deterministic", action="store_true", help="enable deterministic mode")
    parser.add_argument("--amp-opt-level", type=str, choices=["O0", "O1", "O2"], help="deprecated apex amp level")
    parser.add_argument("--output", default="output", type=str, metavar="PATH", help="root output folder")
    parser.add_argument("--name", type=str, help="override model name")
    parser.add_argument("--tag", help="experiment tag")
    parser.add_argument("--eval", action="store_true", help="evaluation only")
    parser.add_argument("--throughput", action="store_true", help="throughput only")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument("--fused_window_process", action="store_true", help="use fused window process")
    parser.add_argument("--fused_layernorm", action="store_true", help="use fused layernorm")
    parser.add_argument("--optim", type=str, help="optimizer override")
    parser.add_argument("--tasks", type=str, default="depth", help="comma separated tasks")
    parser.add_argument("--nyud", type=str, help="NYUD dataset path")
    parser.add_argument("--pascal", type=str, help="PASCAL dataset path")
    parser.add_argument("--eval-training-freq", type=int, help="training-set eval frequency")
    parser.add_argument("--freeze-backbone", action="store_true", help="freeze encoder layers")
    parser.add_argument("--skip_initial_validation", action="store_true", help="skip initial validation")
    parser.add_argument("--decoder_map", type=str, help="decoder head JSON map")
    parser.add_argument("--skip_decoder", action="store_true", help="skip loading decoder weights")
    parser.add_argument("--debug-repro-steps", type=int, default=0, help="debug reproducibility steps")
    args = parser.parse_args()
    config = get_config(args)
    return args, config


if __name__ == "__main__":
    args, config = parse_option()
    result = run_pruning_experiment(config, args)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
