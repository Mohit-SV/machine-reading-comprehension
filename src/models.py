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
from constants import *
from torch import Tensor
from ranger21 import Ranger21
from typing import List, Tuple, Dict, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
import configargparse
from torchmetrics.functional.text.squad import (
    _compute_f1_score,
    _compute_exact_match_score,
)

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
        profiler,
        llmv3_checkpoint=LLMv3_BACKBONE,
        roberta_checkpoint=RoBERTa_BACKBONE
    ):
        """
        :param model_name: LLMv3 or RoBERTa
        :param max_epochs: max epochs till which training can go
        :param num_batches_per_epoch: number of batches that the model sees in 
            each epoch
        :param lr: initial learning rate of the model
        :param use_ReduceLROnPlateau: to use ReduceLROnPlateau or not
        :param patience: patience for ReduceLROnPlateau
        :param optimizer: adam or ranger
        :param log_val_every_n_steps: iteratively tensorboard logging after 
            how many validation steps the model should log
        :param log_test_every_n_steps: iteratively tensorboard logging after 
            how many test steps the model should log
        :param profiler: PyTorch profiler object
        :param llmv3_checkpoint: LLMv3 checkpoint
        :param roberta_checkpoint: RoBERTa checkpoint
        """
        super().__init__()

        self.lr = lr
        self.use_ReduceLROnPlateau = use_ReduceLROnPlateau
        self.patience = patience
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.num_batches_per_epoch = num_batches_per_epoch
        self.log_val_every_n_steps = log_val_every_n_steps
        self.log_test_every_n_steps = log_test_every_n_steps
        self.count_epoch = 0
        self.profiler = profiler

        if model_name == "LLMv3":
            self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(llmv3_checkpoint)
            self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(
                llmv3_checkpoint
            )
            self.forward = self.forward_llmv3
        elif model_name == "RoBERTa":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_checkpoint)
            self.model = RobertaForQuestionAnswering.from_pretrained(roberta_checkpoint)
            self.forward = self.forward_roberta
        else:
            raise ValueError(
                f"Unexpected type of model specified ({self.model_name}), "
                f"choose one of 'LLMv3' or 'RoBERTa'"
            )

        self.stats_keys = ["loss", "f1_score", "em"]

    ###################################################
    # Optimizer
    ###################################################

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

    ###################################################
    # Forward functions
    ###################################################

    def forward_llmv3(
        self, batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]]
    ) -> QuestionAnsweringModelOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            bbox=batch["bbox"],
            pixel_values=batch["pixel_values"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )

    def forward_roberta(
        self, batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]]
    ) -> QuestionAnsweringModelOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )

    ###################################################
    # Training
    ###################################################

    def training_step(
        self, batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]]
    ) -> Dict[str, Tensor]:
        """Runs forward function and logs loss + metrics in training step"""
        outputs = self.forward(batch)
        return self.get_stats_each_step(batch, outputs, "train")

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        """
        Computes and logs averages of training step losses and metrics at end of each epoch
        """
        # outputs.loss = [{'loss': tensor(6.2211)}, {'loss': tensor(6.1990)}...]
        self.run_for_each_epoch(outputs, "train")
        self.profiler.step()

    ###################################################
    # Validation
    ###################################################

    def validation_step(
        self,
        batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """Runs forward function and logs loss + metrics in validation step"""
        outputs = self.forward(batch)
        return self.get_stats_each_step(batch, outputs, "val", batch_idx)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        """
        Computes and logs averages of validation step losses and metrics at end of each epoch
        """
        self.run_for_each_epoch(outputs, "val")
        self.count_epoch += 1
        self.profiler.step()

    ###################################################
    # Testing
    ###################################################

    def test_step(
        self,
        batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """Runs forward function and logs loss + metrics in testing step"""
        outputs = self.forward(batch)
        return self.get_stats_each_step(batch, outputs, "test", batch_idx)

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]):
        """
        Logs average losses + metrics over all test steps in console
        """
        # gathers stats from 1 epoch
        final_stats = {
            stat: torch.stack([step_out[stat] for step_out in outputs]).mean()
            for stat in self.stats_keys
        }
        logger.info(f"test stats: {final_stats}")
        self.profiler.step()

    ###################################################
    # Shared functions
    ###################################################

    def get_stats_each_step(
        self,
        batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]],
        outputs: QuestionAnsweringModelOutput,
        mode: str,
        batch_idx: Union[int, None] = None,
    ) -> Dict[str, Tensor]:
        """
        Logs average losses + metrics over all steps in an epoch of train/validation in tensorboard

        :param batch: input batch
        :param mode: "train" or "val"
        :param batch_idx: batch index
        :return: statistics (loss + metrics) of each step
        """
        target_answers, predicted_answers = self.decode_output(batch, outputs)

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

        :param outputs: list of stat dicts from each step
        :param mode: "train" or "val"
        """
        epoch_end_stats = {}

        epoch_end_stats = {
            stat: round(torch.stack(
                [
                    step_out[stat]
                    for step_out in outputs
                    if not torch.isnan(step_out[stat])
                ]
            ).mean().item(), 4)
            for stat in self.stats_keys
        }

        logger.info(
            f"[Epoch {self.count_epoch}] {mode} stats: {epoch_end_stats}" 
        )

        log_stats = {
            f"{mode}_{stat}_per_epoch": value for stat, value in epoch_end_stats.items()
        }

        self.log_dict(log_stats, on_epoch=True, on_step=False)

    ###################################################
    # Metrics
    ###################################################

    def get_all_scores(
        self, target_answers: List[str], predicted_answers: List[str]
    ) -> Dict[str, Tensor]:
        """
        Gets metrics: 'f1_score', 'em'.

        :target_answers: list of gold standard answers
        :predicted_answers: list of answers predicted by the model
        :return: dictionary containing metrics for respective target_answers, predicted_answers
        """

        batch_em_score = (
            torch.stack(
                [
                    _compute_exact_match_score(target_answer, predicted_answer)
                    for target_answer, predicted_answer in zip(
                        target_answers, predicted_answers
                    )
                ]
            )
            .float()
            .mean()
        )

        batch_f1_score = (
            torch.stack(
                [
                    _compute_f1_score(target_answer, predicted_answer)
                    for target_answer, predicted_answer in zip(
                        target_answers, predicted_answers
                    )
                ]
            )
            .float()
            .mean()
        )

        return {"f1_score": batch_f1_score, "em": batch_em_score}

    ###################################################
    # Post-processing
    ###################################################

    def decode_output(
        self,
        batch: Dict[str, Union[List[Union[str, int, Tensor]], Tensor]],
        outputs: Dict[str, Tensor],
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
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_pos_confidences, pred_answer_start_indices = start_logits.max(dim=-1)
        end_pos_confidences, pred_answer_end_indices = end_logits.max(dim=-1)
        answer_confidences = (start_pos_confidences + end_pos_confidences) / 2

        pred_batch_answer_tokens = [
            input_ids.squeeze(-1)[answer_start_index : answer_end_index + 1]
            for answer_start_index, answer_end_index, input_ids in zip(
                pred_answer_start_indices, pred_answer_end_indices, batch["input_ids"]
            )
        ]
        # +1 : to include last character

        # answers from each span of each sample
        predicted_span_answers = [
            (
                self.tokenizer.decode(
                    answer_tokens,
                    skip_special_tokens=True,
                    # clean_up_tokenization_spaces=False # https://discuss.huggingface.co/t/layoutlmv3-q-a-inference/29872/3
                )
            ).replace("\n", "")
            for answer_tokens in pred_batch_answer_tokens
        ]

        # answers for each question/sample
        predicted_answers = self.get_predicted_answers(
            predicted_span_answers,
            answer_confidences,
            batch["overflow_to_sample_mapping"],
        )
        target_answers = batch["answer_text"]

        return target_answers, predicted_answers

    @staticmethod
    def get_predicted_answers(
        predicted_span_answers: List[str],
        answer_confidences: Tensor,
        sample_mapping: List[int],
    ):
        """
        Returns predicted answers of each sample from given predicted answers of
        each span of every sample

        :param predicted_span_answers: list of answers predicted in each span of
            every sample
        :param answer_confidences: list of likelihoods of answers from respective
            spans being correct
        :param sample_mapping: span index to sample index mapping
        :return: list of predicted answers corresponding to given predicted_span_answers
        """
        predicted_answers = {}

        prev_span_index = 0
        for span_index, answer in enumerate(predicted_span_answers):
            sample_index = sample_mapping[span_index]
            if sample_index in predicted_answers:
                if len(predicted_answers[sample_index]) != 0:
                    if (
                        len(answer) != 0
                        and answer_confidences[span_index]
                        > answer_confidences[prev_span_index]
                    ):
                        predicted_answers[sample_index] = answer
                        prev_span_index = span_index
                else:
                    predicted_answers[sample_index] = answer
                    prev_span_index = span_index
            else:
                predicted_answers[sample_index] = answer
                prev_span_index = span_index

        return list(predicted_answers.values())

    ###################################################
    # Arguments
    ###################################################

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
