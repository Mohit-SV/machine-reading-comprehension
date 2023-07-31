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
)
from pytorch_lightning.loggers import TensorBoardLogger
from parser_utils import get_args
import logging
import os
from copy import deepcopy
import configargparse
from constants import *
import time
import datetime
from pytorch_lightning.profiler import PyTorchProfiler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)
# logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

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
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        tokenizer_max_length=args.tokenizer_max_length,
        tokenizer_stride=args.tokenizer_stride,
        num_workers=args.num_workers,
        is_layout_dependent=args.is_layout_dependent,
        include_references=args.include_references,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    shuffle_trainset = False if args.dataloader_seed is None else True

    train_dataloader = dataset.to_dataloader("train", shuffle_trainset)
    eval_dataloader = dataset.to_dataloader("dev")
    test_dataloader = dataset.to_dataloader("test")

    # needed for initializing the Ranger21 optimizer
    num_train_batches_per_epoch = len(train_dataloader)

    ##################################################
    # Instantiate model
    ##################################################

    # initializing appropriate variables for lr: constant/scheduler/auto
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
        raise ValueError(
            "The LR scheduler mode is undefined, it should be one of [0, 1, 2]"
        )

    if args.optimizer.lower() == "ranger" and args.lr_scheduler_mode:
        raise AssertionError(
            "While using 'ranger' optimizer, external learning schedulers cannot be used."
        )
    
    with torch.profiler.profile(
        # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./pytorch_profiler_outs/{args.experiment_name}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

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
            profiler=prof
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
        runs_dir = os.path.join(SRC_DIRECTORY, "runs")
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs(runs_dir, exist_ok=True)

        early_stopping = EarlyStopping(
            monitor="val_loss_per_epoch",
            mode="min",
            verbose=True,
            patience=args.early_stopping_patience,
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
            save_dir=runs_dir,
            version=None,
            name=args.experiment_name,
        )

        # profiler = PyTorchProfiler(
        #     dirpath='profiler_output', 
        #     filename=args.experiment_name, 
        #     export_to_chrome=True, 
        #     row_limit=None, 
        #     sort_by_key='cpu_memory_usage', 
        #     record_module_names=True,
        #     profile_memory=True,
        #     # record_functions=[ # profiled_functions
        #     #     '[LightningModule]QAModel.training_step', 
        #     #     '[LightningModule]QAModel.training_epoch_end',
        #     #     '[LightningModule]QAModel.validation_step',
        #     #     '[LightningModule]QAModel.validation_epoch_end',
        #     #     '[LightningModule]QAModel.test_step',
        #     #     '[LightningModule]QAModel.test_epoch_end'
        #     #     ]
        #     )

        callbacks = [lr_logger, early_stopping, checkpoint_callback]

        trainer = pl.Trainer.from_argparse_args(
            args,
            log_every_n_steps=args.log_train_every_n_steps,
            callbacks=callbacks,
            enable_checkpointing=checkpoint_callback,
            logger=[tb_logger],
            accelerator=accelerator,
            max_epochs=args.max_epochs,
            default_root_dir=model_save_dir,
            benchmark=True,  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
            auto_lr_find=auto_lr_find,  # sets the suggested learning rate in self.lr
            # profiler=profiler # "simple"
            # enable_progress_bar=False,
        )

        ##################################################
        # Execute
        ##################################################

        start_time = time.time()
        trainer.fit(model, train_dataloader, eval_dataloader)
        train_ended_time = time.time()
        logger.info(
            f"Time taken by trainer.fit (training + validation): {datetime.timedelta(seconds = train_ended_time - start_time)}"
        )

        trainer.test(model, dataloaders=test_dataloader, verbose=False, ckpt_path="best")
        test_ended_time = time.time()
        logger.info(
            f"Time taken by trainer.test (testing): {datetime.timedelta(seconds = test_ended_time - train_ended_time)}"
        )


    # Useful terminal commands to view tensorboard visualizations:
    # !tensorboard --logdir=src\runs --bind_all
    # !tensorboard --logdir_spec=roberta:src\runs\test\version_1,llmv3:src\runs\test\version_2 --bind_all

    # !tensorboard --logdir=./pytorch_profiler_outs --bind_all

    # Windows:: For anyone else looking for a workaround:
    # Change line 114 in '\path_to_python_installation\Lib\site-packages\torch_tb_profiler\profiler\data.py'
    # to
    # trace_json = json.loads(data.replace(b"\\", b"\\\\"), strict=False)
    # This works for me as a dirty workaround
