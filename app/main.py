# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import multiprocessing as mp
import pprint
import sys
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="If specified, will not spin up new processes. "
    "The main code runs the main process, which makes it easier to \
    debug with checkpointing.",
)


def process_main(rank, fname, world_size, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config. ${VAR} references (e.g. the ROBO_JEPA_* paths set by truba/robo_jepa_train.slurm) are expanded
    # from the environment; unset variables are left as they are.
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(os.path.expandvars(y_file.read()), Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine). The port is offset by the Slurm job id so
    # jobs sharing a node do not collide on it.
    world_size, rank = init_distributed(
        port=37249 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000,
        rank_and_world_size=(rank, world_size),
        timeout=datetime.timedelta(minutes=60),
    )
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        process_main(rank=0, fname=args.fname, world_size=1, devices=["cuda:0"])
    else:
        num_gpus = len(args.devices)
        mp.set_start_method("spawn")
        procs = [
            mp.Process(target=process_main, args=(rank, args.fname, num_gpus, args.devices)) for rank in range(num_gpus)
        ]
        for p in procs:
            p.start()
        # Exit non-zero if any rank failed, so a batch job reports the failure.
        for p in procs:
            p.join()
        sys.exit(next((p.exitcode for p in procs if p.exitcode), 0))
