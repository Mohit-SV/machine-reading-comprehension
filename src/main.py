"""
Main project file to run experiments
"""

import torch
import pytorch_lightning as pl
from dataloader import SquadDataset
from models import QAModel
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    # LearningRateFinder
)
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler
from parser_utils import get_args
import logging
import os
from copy import deepcopy
import configargparse
from constants import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    ##################################################
    # Collect main args
    ##################################################

    args, parser, accelerator = get_args()

    parser = pl.Trainer.add_argparse_args(parent_parser=parser, use_argument_group=True)
    parser = SquadDataset.add_argparse_args(parser)
    parser = QAModel.add_argparse_args(parser)

    # Workaround to parse args from config file twice
    _actions = deepcopy(parser._actions)
    _option_string_actions = deepcopy(parser._option_string_actions)
    parser = configargparse.ArgParser(default_config_files=["parameters.ini"])
    parser._actions = _actions
    parser._option_string_actions = _option_string_actions
    # End workaround

    args, unknown = parser.parse_known_args()

    logger.info(f"parsed args: {args}")
    if len(unknown) > 0:
        logger.info(f"Unknown args: {unknown}")

    ##################################################
    # Instantiate data loader
    ##################################################

    pl.seed_everything(args.dataloader_seed, workers=True)

    dataset = SquadDataset(
        model_name=args.model_name,
        batch_size=args.batch_size,
        dev_ratio=args.dev_split,
        test_ratio=args.train_split,
        seed=args.split_seed,
        llmv3_max_length=args.llmv3_max_length,
        roberta_max_length=args.roberta_max_length,
        llmv3_stride=args.llmv3_stride,
        roberta_stride=args.roberta_stride,
        num_workers=args.num_workers
    )

    shuffle_dataloader = False if args.dataloader_seed is None else True

    train_dataloader = dataset.dataloader("train", shuffle_dataloader)
    eval_dataloader = dataset.dataloader("dev", shuffle_dataloader)
    test_dataloader = dataset.dataloader("test", shuffle_dataloader)

    # needed to properly configure the Ranger21 optimizer
    num_train_batches_per_epoch = len(train_dataloader)

    ##################################################
    # Instantiate model
    ##################################################

    # lr - constant/scheduler/auto
    if args.lr_scheduler_mode == 0:
        use_ReduceLROnPlateau = False
        auto_lr_find = False
    elif args.lr_scheduler_mode == 1:
        use_ReduceLROnPlateau = False
        auto_lr_find = True
    elif args.lr_scheduler_mode == 2:
        use_ReduceLROnPlateau = True
        auto_lr_find = False
    else:
        ValueError()

    if args.optimizer.lower()=='ranger' and args.lr_scheduler_mode:
        raise AssertionError(
                "While using 'ranger' optimizer, external learning schedulers cannot be used."
            )

    model = QAModel(
        model_name=args.model_name, 
        num_batches_per_epoch=num_train_batches_per_epoch,
        max_epochs=args.max_epochs, 
        lr=args.lr,
        use_ReduceLROnPlateau=use_ReduceLROnPlateau,
        patience=args.reduce_lr_on_plateau_patience,
        optimizer=args.optimizer, 
        log_val_every_n_steps=args.log_val_every_n_steps,
        log_test_every_n_steps=args.log_test_every_n_steps,
        normalize=False,
        )

    if args.checkpoint is not None:
        logger.info("Loading model checkpoint from {args.checkpoint}")
        checkpoint = torch.load(
            args.checkpoint, map_location=lambda storage, loc: storage
        )
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        except RuntimeError:
            logger.warning(
                "Found a mismatch in sizes of the initialized and the pre-trained model"
            )

    ##################################################
    # Initialize trainer
    ##################################################

    model_save_dir = os.path.join(SRC_DIRECTORY, "models", args.experiment_name)
    os.makedirs(model_save_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss_per_epoch", mode="min", verbose=True, patience=args.early_stopping_patience
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        save_top_k=1,
        verbose=True,
        monitor="val_loss_per_epoch",
        mode="min",
        auto_insert_metric_name=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(SRC_DIRECTORY, "runs"), 
        version=None, 
        name=args.experiment_name
    )
    callbacks = [lr_logger, early_stopping, checkpoint_callback]

    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("runs/profiler0"),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    # ) # active: number of profiler steps

    trainer = pl.Trainer.from_argparse_args(
        args,
        log_every_n_steps=args.log_train_every_n_steps,
        callbacks=callbacks,
        enable_checkpointing=checkpoint_callback,
        logger=[tb_logger],
        accelerator='auto',
        max_epochs=args.max_epochs,
        default_root_dir=model_save_dir,
        benchmark=True,  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        auto_lr_find=auto_lr_find, # sets the suggested learning rate in self.lr
        profiler="simple" # profiler is not working
    )

    ##################################################
    # Execute
    ##################################################

    trainer.fit(model, train_dataloader, eval_dataloader)

    trainer.test(
        model, 
        dataloaders=test_dataloader, 
        verbose=False, 
        ckpt_path="best" 
    )

    # !tensorboard --logdir=src\runs --bind_all
    # !tensorboard --logdir_spec=roberta:src\runs\test\version_1,llmv3:src\runs\test\version_2 --bind_all
