import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ag_mtlora.stage1 import create_stage1_logger, replay_stage1_partition_search


def parse_args():
    parser = argparse.ArgumentParser("AG-MTLoRA Stage-1 partition search replay")
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to the original base config file")
    parser.add_argument("--stage1-dir", type=str, required=True, help="existing Stage-1 output directory")
    parser.add_argument(
        "--score-source",
        type=str,
        default="group_proxy",
        choices=["final_predictions", "group_proxy"],
        help="which saved score artifact to use for partition search replay",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stage1_dir = os.path.abspath(args.stage1_dir)
    logger = create_stage1_logger(stage1_dir)
    logger.info(
        "Running AG-MTLoRA Stage-1 replay search | stage1_dir=%s | score_source=%s | cfg=%s",
        stage1_dir,
        args.score_source,
        os.path.abspath(args.cfg),
    )

    artifacts = replay_stage1_partition_search(
        base_cfg_path=os.path.abspath(args.cfg),
        stage1_dir=stage1_dir,
        search_score_source=args.score_source,
        logger=logger,
    )
    logger.info("AG-MTLoRA Stage-1 replay search finished.")
    logger.info(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
