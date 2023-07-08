"""
Loads all the arguments
"""

import configargparse
import torch
from typing import Tuple
from constants import SRC_DIRECTORY
import os


def get_args() -> Tuple[configargparse.Namespace, configargparse.ArgParser, str]:
    """
    Imports arguments from command line and/or parameters.ini config file
        
    :returns:
        - args: Namespace object containing [Main] args
        - parser: configargparse parser object with [Main] args
        - device: device (cpu/cuda) on which dataloader runs
    """
    parser = configargparse.ArgParser(default_config_files=[os.path.join(SRC_DIRECTORY,"parameters.ini")])
    parser.add_argument(
        "-c",
        "--my-config",
        default=os.path.join(SRC_DIRECTORY,"parameters.ini"),
        is_config_file=True,
        help="config file path",
    )
    parser.add_argument(
        "--experiment_name",
        help="Experiment name.",
    )
    parser.add_argument(
        "--model_name",
        help="Model to be used: LLMv3 or RoBERTa.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument("--early_stopping_patience", type=int, help="patience in early stopping")

    args, unknown = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    accelerator = "cuda" if args.cuda else "cpu"

    return args, parser, accelerator