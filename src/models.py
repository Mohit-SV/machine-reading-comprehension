# this is where models are defined

import pytorch_lightning as pl
import torch
from transformers import (
    RobertaForQuestionAnswering,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3TokenizerFast,
    RobertaTokenizerFast,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import logging
import numpy as np
import string, re
from constants import *
from torch import Tensor
from ranger21 import Ranger21
from typing import List, Tuple, Dict, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
import configargparse

# Init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


class QAModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        max_epochs,
        num_batches_per_epoch,
        lr: float,
        use_ReduceLROnPlateau,
        patience,
        optimizer,
        log_val_every_n_steps,
        log_test_every_n_steps,
        llmv3_checkpoint=LLMv3_BACKBONE,
        roberta_checkpoint=RoBERTa_BACKBONE,
        normalize: bool = False,
    ):
        """

        Args:
            model_name: name or path to load model from
        """
        super().__init__()

        self.lr = lr
        self.use_ReduceLROnPlateau = use_ReduceLROnPlateau
        self.patience = patience
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.num_batches_per_epoch = num_batches_per_epoch
        self.normalize = normalize
        self.log_val_every_n_steps = log_val_every_n_steps
        self.log_test_every_n_steps = log_test_every_n_steps

        self.model_name = model_name
        if self.model_name == "LLMv3":
            self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(llmv3_checkpoint)
            self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(
                llmv3_checkpoint
            )
            self.forward = self.forward_llmv3
        elif self.model_name == "RoBERTa":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_checkpoint)
            self.model = RobertaForQuestionAnswering.from_pretrained(roberta_checkpoint)
            self.forward = self.forward_roberta
        else:
            raise ValueError(
                f"Unexpected type of model specified ({self.model_name}), "
                f"choose one of 'LLMv3' or 'RoBERTa'"
            )

        self.stats_keys = ["loss", "f1_score", "precision", "recall", "em"]
        # updated cumulatively for each epoch by aggregating (sum) scores from each step
        self.cumulative_stats = {
            mode: {key: 0 for key in self.stats_keys} for mode in ["train", "val"]
        }

    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            if self.use_ReduceLROnPlateau:
                scheduler = ReduceLROnPlateau(
                    optimizer, mode="min", patience=self.patience
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val_loss_per_step(s)",
                }
        elif self.optimizer.lower() == "ranger":
            optimizer = Ranger21(
                self.parameters(),
                lr=self.lr,
                num_epochs=self.max_epochs,
                num_batches_per_epoch=self.num_batches_per_epoch,
                use_warmup=True,
                warmdown_active=True,
                warmdown_min_lr=1e-5,
                weight_decay=1e-4,
            )
        else:
            raise AssertionError(
                "Currently only 'ranger' and 'adam' are the supported optimizers."
            )
        return optimizer

    def forward_llmv3(self, batch: Dict) -> QuestionAnsweringModelOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            pixel_values=batch["pixel_values"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )

    def forward_roberta(self, batch: Dict) -> QuestionAnsweringModelOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )

    ##################################
    # Training
    ##################################

    def training_step(
        self, batch: Dict[str, Union[List[Union[str, Tensor]], Tensor]]
    ) -> Dict[str, Tensor]:
        """Runs forward function and logs loss + metrics in training step"""
        return self.run_for_each_step(batch, "train")

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        """
        Computes and logs averages of training step losses and metrics at end of each epoch
        """
        # outputs.loss = [{'loss': tensor(6.2211)}, {'loss': tensor(6.1990)}...]
        self.run_for_each_epoch(outputs, "train")

    def on_train_end(self):
        self.run_at_end("train")

    ##################################
    # Validation
    ##################################

    def validation_step(
        self, batch: Dict[str, Union[List[Union[str, Tensor]], Tensor]], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Runs forward function and logs loss + metrics in validation step"""
        return self.run_for_each_step(batch, "val", batch_idx)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        """
        Computes and logs averages of validation step losses and metrics at end of each epoch
        """
        self.run_for_each_epoch(outputs, "val")

    def on_validation_end(self):
        self.run_at_end("val")

    ##################################
    # Testing
    ##################################

    def test_step(
        self, batch: Dict[str, Union[List[Union[str, Tensor]], Tensor]], batch_idx: int
    ) -> Dict[str, Tensor]:
        """Runs forward function and logs loss + metrics in testing step"""
        return self.run_for_each_step(batch, "test", batch_idx)

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        """
        Logs average losses + metrics over all test steps in console
        """
        # gathers stats from 1 epoch
        final_stats = {
            stat: torch.stack([step_out[stat] for step_out in outputs]).mean()
            for stat in self.stats_keys
        }
        logging.info(f"test stats: {final_stats}")

    ##################################
    # Shared functions
    ##################################

    def run_for_each_step(
        self,
        batch: Dict[str, Union[List[Union[str, Tensor]], Tensor]],
        mode: str,
        batch_idx: Union[int, None] = None,
    ) -> Dict[str, Tensor]:
        """
        Logs average losses + metrics over all steps in an epoch of train/validation in tensorboard

        :param mode: "train" or "val"
        :return: statistics (loss + metrics) of each step
        """
        outputs = self.forward(batch)
        target_answers, predicted_answers = self.decode_output(batch, outputs)

        if self.normalize:
            target_answers = [
                self.normalize_text(target_answer) for target_answer in target_answers
            ]
            predicted_answers = [
                self.normalize_text(predicted_answer)
                for predicted_answer in predicted_answers
            ]

        stats = self.get_all_scores(target_answers, predicted_answers)
        stats["loss"] = outputs.loss

        log_stats = {
            f"{mode}_{stat}_per_step(s)": value for stat, value in stats.items()
        }

        # pytorch_lightning's trainer parameter log_every_n_steps sets how
        # frequently it should log training step stats on tensorboard
        if mode == "train":
            self.log_dict(log_stats, on_epoch=False, on_step=True)
        # custom code to set how frequently it should log validation step stats on tensorboard
        elif mode == "val":
            if (batch_idx + 1) % self.log_val_every_n_steps:
                self.log_dict(log_stats, on_epoch=False, on_step=True)
        # custom code to set how frequently it should log testing step stats on tensorboard
        else:
            if (batch_idx + 1) % self.log_test_every_n_steps:
                self.log_dict(log_stats, on_epoch=False, on_step=True)

        return stats

    def run_for_each_epoch(self, outputs: List[Dict[str, Tensor]], mode: str):
        """
        Logs average losses + metrics over all steps in an epoch of train/validation on tensorboard

        :outputs: list of stat dicts from each step
        :param mode: "train" or "val"
        """
        epoch_end_stats = {}

        for stat in self.stats_keys:
            avg_stat = torch.stack(
                [
                    step_out[stat]
                    for step_out in outputs
                    if not torch.isnan(step_out[stat])
                ]
            ).mean()
            epoch_end_stats[f"{mode}_{stat}_per_epoch"] = avg_stat
            self.cumulative_stats[mode][stat] = torch.add(
                avg_stat, self.cumulative_stats[mode][stat]
            )

        self.log_dict(epoch_end_stats, on_epoch=True, on_step=False)

    def run_at_end(self, mode: str):
        """
        Logs average losses + metrics over all epochs of train/validation on console

        :param mode: "train" or "val"
        """
        final_stats = {
            stat: torch.div(self.cumulative_stats[mode][stat], self.max_epochs)
            for stat in self.stats_keys
        }
        logging.info(f"{mode} stats: {final_stats}")

    ##################################
    # Metrics
    ##################################

    def get_all_scores(
        self, target_answers: List[str], predicted_answers: List[str]
    ) -> Dict[str, Tensor]:
        """
        Gets metrics: 'f1_score', 'precision', 'recall', 'em'.

        :target_answers: list of gold standard answers
        :predicted_answers: list of answers predicted by the model
        :return: dictionary containing metrics for respective target_answers, predicted_answers
        """

        def get_scores_per_sample(
            pred_tokens: List[str], truth_tokens: List[str]
        ) -> List[float]:
            precision, recall, f1_score = 0, 0, 0
            if len(pred_tokens) == 0 or len(truth_tokens) == 0:
                f1_score = int(pred_tokens == truth_tokens)
            else:
                n_common_tokens = len(set(pred_tokens) & set(truth_tokens))
                if n_common_tokens != 0:
                    precision = n_common_tokens / len(pred_tokens)
                    recall = n_common_tokens / len(truth_tokens)
                    f1_score = 2 * (precision * recall) / (precision + recall)
            return [precision, recall, f1_score]

        scores = np.mean(
            [
                get_scores_per_sample(target_answer.split(), predicted_answer.split())
                for target_answer, predicted_answer in zip(
                    target_answers, predicted_answers
                )
            ],
            axis=0,
        )

        return {
            "precision": torch.tensor(scores[0]),
            "recall": torch.tensor(scores[1]),
            "f1_score": torch.tensor(scores[2]),
            "em": torch.tensor(self.get_em_score(target_answers, predicted_answers)),
        }

    @staticmethod
    def get_em_score(target_answers: List[str], predicted_answers: List[str]) -> float:
        """
        Outputs exact-match (em) scores.

        :target_answers: list of gold standard answers
        :predicted_answers: list of answers predicted by the model
        :return: list of exact match scores for respective target_answers, predicted_answers
        """
        return np.mean(
            [
                int(target_answer == predicted_answer)
                for target_answer, predicted_answer in zip(
                    target_answers, predicted_answers
                )
            ]
        )

    ##################################
    # Post-processing
    ##################################

    def decode_output(
        self, batch: Dict[str, Tensor], outputs: QuestionAnsweringModelOutput
    ) -> Tuple[List[str]]:
        """
        Decodes model outputs into predicted answers.

        :param batch: input batch from dataloader
        :param outputs: output from respective (train/val/test) step
        :returns:
            - target_answers: list of gold standard answers in the batch
            - predicted_answers: list of predicted answers for questions in the batch
        """
        # output start_logits and end_logits indicate which token the model thinks
        # is at the start of the answer, and which token is at the end of the answer
        pred_answer_start_indices = outputs.start_logits.argmax(-1)
        pred_answer_end_indices = outputs.end_logits.argmax(-1)

        pred_batch_answer_tokens = [
            input_ids.squeeze(-1)[answer_start_index : answer_end_index + 1]
            for answer_start_index, answer_end_index, input_ids in zip(
                pred_answer_start_indices, pred_answer_end_indices, batch["input_ids"]
            )
        ]
        # +1 : to include last character

        predicted_answers = [
            self.tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
                # clean_up_tokenization_spaces=False # https://discuss.huggingface.co/t/layoutlmv3-q-a-inference/29872/3
            )
            for answer_tokens in pred_batch_answer_tokens
        ]

        target_answers = batch["answer_text"]

        return target_answers, predicted_answers

    @staticmethod
    def normalize_text(text: str) -> str:
        """1. Lowercases text 2. Removes articles, punctuations 3. Fixes white spaces"""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(text))))

    ##################################
    # Arguments
    ##################################

    @staticmethod
    def add_argparse_args(
        parent_parser: configargparse.ArgParser,
    ) -> configargparse.ArgParser:
        parser = parent_parser.add_argument_group("QAModel")
        parser.add_argument(
            "--lr",
            type=float,
            default=2e-5,
            help="To include citation numbers in text or not.",
        )
        parser.add_argument(
            "--lr_scheduler_mode",
            type=int,
            default=1,
            help="scheduler_mode = {0: None, 1: 'lighting-auto-lr-finder', 2: 'ReduceLROnPlateau'}",
        )
        parser.add_argument("--checkpoint", type=str, help="Saved model checkpoints")
        parser.add_argument("--optimizer", type=str, help="optimizer: adam or ranger")
        parser.add_argument(
            "--log_train_every_n_steps",
            type=int,
            default=1,
            help="After how many batches train step results have to be logged.",
        )
        parser.add_argument(
            "--log_val_every_n_steps",
            type=int,
            default=1,
            help="After how many batches val step results have to be logged.",
        )
        parser.add_argument(
            "--log_test_every_n_steps",
            type=int,
            default=1,
            help="After how many batches test step results have to be logged.",
        )
        parser.add_argument(
            "--reduce_lr_on_plateau_patience",
            type=int,
            help="patience for reduce_lr_on_plateau",
        )
        return parent_parser
