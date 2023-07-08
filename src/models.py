# this is where models are defined

import pytorch_lightning as pl
import torch
from transformers import (
    RobertaForQuestionAnswering,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3TokenizerFast,
    RobertaTokenizerFast,
)
import logging
import numpy as np
import string, re
from constants import *
from torch import Tensor
from ranger21 import Ranger21
from typing import List, Optional, Dict
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Init logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s"
)


class QAModel(pl.LightningModule):
    def __init__(self, 
                 model_name: str,   
                 max_epochs,
                 num_batches_per_epoch,
                 lr: float ,
                 use_ReduceLROnPlateau, 
                 patience,
                 optimizer,
                 log_val_every_n_steps,
                 log_test_every_n_steps,          
                 llmv3_checkpoint = LLMv3_BACKBONE,
                 roberta_checkpoint = RoBERTa_BACKBONE,
                 normalize: bool = False):
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
            # sequence_length = token_sequence_length + patch_sequence_length + 1 where 1 is for [CLS]
        elif self.model_name == "RoBERTa":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_checkpoint)
            self.model = RobertaForQuestionAnswering.from_pretrained(roberta_checkpoint)
            # sequence_length = token_sequence_length + 1 where 1 is for [CLS]
        else:
            raise ValueError()
        self.forward = {
            "LLMv3": self.forward_LLMv3,
            "RoBERTa": self.forward_RoBERTa,
        }

        self.stats_keys = ['loss', 'f1_score', 'precision', 'recall', 'em']
        self.cumulative_stats = {
            mode : {key : 0 for key in self.stats_keys}
            for mode in ['train', 'val']
            }
        self.val_cumulative_stats = {key:0 for key in self.stats_keys}
        self.test_cumulative_stats = {key:0 for key in self.stats_keys}
        
    def configure_optimizers(self):     
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, patience=self.patience)
            if self.use_ReduceLROnPlateau:
                scheduler = ReduceLROnPlateau(optimizer, 'min')
                return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_per_step(s)"}
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
                "Currently only 'ranger' and 'adam' are supported as optimizer."
            )
        return optimizer

    def forward_LLMv3(self, batch):
        return self.model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            bbox=batch['bbox'], 
            pixel_values=batch['pixel_values'], 
            start_positions=batch['start_positions'], 
            end_positions=batch['end_positions']
            )
    
    def forward_RoBERTa(self, batch):
        return self.model(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            start_positions=batch['start_positions'], 
            end_positions=batch['end_positions']
            )

    def training_step(self, batch):
        return self.run_for_each_step(batch, 'train')
    
    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """
        Computes average training loss
        :param outputs: outputs after every epoch end
        :return: output - average training loss
        """
        # outputs.loss = [{'loss': tensor(6.2211)}, {'loss': tensor(6.1990)}...]
        self.run_for_each_epoch(outputs, 'train')

    def on_train_end(self):
        self.run_at_end('train')

    def validation_step(self, batch, batch_idx):
        return self.run_for_each_step(batch, 'val', batch_idx)
    
    def validation_epoch_end(self, outputs: List[Tensor]):
        """
        Computes average validation loss
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        self.run_for_each_epoch(outputs, 'val')

    def on_validation_end(self):
        self.run_at_end('val')

    def test_step(self, batch, batch_idx):
        return self.run_for_each_step(batch, 'test', batch_idx)
    
    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """
        Computes average test loss and metrics
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        # runs for 1 epoch
        final_stats = {
            stat : torch.stack([step_out[stat] for step_out in outputs]).mean()
            for stat in self.stats_keys
        }
        logging.info(f"test stats: {final_stats}")

    def run_for_each_step(self, batch, mode: str, batch_idx=None):
        outputs = self.forward[self.model_name](batch)
        target_answers, predicted_answers = self.decode_output(batch, outputs)

        if self.normalize:
            target_answers = [self.normalize_text(target_answer) for target_answer in target_answers]
            predicted_answers = [self.normalize_text(predicted_answer) for predicted_answer in predicted_answers]
        
        stats = self.get_all_scores(target_answers, predicted_answers)
        stats['loss'] = outputs.loss

        log_stats = {f'{mode}_{stat}_per_step(s)': value for stat, value in stats.items()}

        if mode=='train':
            self.log_dict(log_stats, on_epoch=False, on_step=True)
        elif mode=='val':
            if (batch_idx + 1) % self.log_val_every_n_steps:
                self.log_dict(log_stats, on_epoch=False, on_step=True)
        else:
            if (batch_idx + 1) % self.log_test_every_n_steps:
                self.log_dict(log_stats, on_epoch=False, on_step=True)

        return stats

    def run_for_each_epoch(self, outputs, mode):
        epoch_end_stats = {}

        for stat in self.stats_keys:
            avg_stat = torch.stack([step_out[stat] for step_out in outputs if not torch.isnan(step_out[stat])]).mean()
            if mode == 'test':
                epoch_end_stats[f"{mode}_{stat}"] = avg_stat
            else:
                self.cumulative_stats[mode][stat] = torch.add(avg_stat, self.cumulative_stats[mode][stat])
                epoch_end_stats[f"{mode}_{stat}_per_epoch"] = avg_stat

        self.log_dict(epoch_end_stats, on_epoch=True, on_step=False)

    def run_at_end(self, mode):
        final_stats = {
            stat: torch.div(self.cumulative_stats[mode][stat], self.max_epochs)
            for stat in self.stats_keys
            }
        logging.info(f"{mode} stats: {final_stats}")
    
    def decode_output(self, batch, outputs):
        # output start_logits and end_logits, indicating which token the model thinks 
        # is at the start of the answer, and which token is at the end of the answer
        pred_answer_start_indices = outputs.start_logits.argmax(-1)
        pred_answer_end_indices = outputs.end_logits.argmax(-1)

        pred_batch_answer_tokens = [
            input_ids.squeeze(-1)[answer_start_index : answer_end_index + 1]
            for answer_start_index, answer_end_index, input_ids in zip(
                pred_answer_start_indices, pred_answer_end_indices, batch['input_ids']
            )
        ]
        # +1 = include last character
        
        predicted_answers = [
            self.tokenizer.decode(
            answer_tokens, 
            skip_special_tokens=True, 
            # clean_up_tokenization_spaces=False # https://discuss.huggingface.co/t/layoutlmv3-q-a-inference/29872/3
            ) 
            for answer_tokens in pred_batch_answer_tokens
        ]
        
        return batch['answer_text'], predicted_answers

    def get_all_scores(self, target_answers, predicted_answers):
        def get_scores_per_sample(pred_tokens, truth_tokens):
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
            ], axis=0
        )

        return {
            "precision": torch.tensor(scores[0]),
            "recall": torch.tensor(scores[1]),
            "f1_score": torch.tensor(scores[2]),
            "em": torch.tensor(self.get_em_score(target_answers, predicted_answers))
        }

    @staticmethod
    def get_em_score(target_answers, predicted_answers):
        return np.mean(
            [
                int(target_answer == predicted_answer)
                for target_answer, predicted_answer in zip(
                    target_answers, predicted_answers
                )
            ]
        )

    @staticmethod
    def normalize_text(text):        
        def remove_articles(text) :
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex," ", text)
        
        def white_space_fix(text) :
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text) :
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(text))))
    
    @staticmethod
    def add_argparse_args(parent_parser):
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
            help="scheduler_mode = {0: None, 1: 'lighting-auto-lr-finder', 2: 'ReduceLROnPlateau'}"
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            help="Saved model checkpoints"
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            help="optimizer: adam or ranger"
        )
        parser.add_argument(
            "--log_train_every_n_steps",
            type=int,
            default=1,
            help="After how many batches train step results have to be logged."
        )
        parser.add_argument(
            "--log_val_every_n_steps",
            type=int,
            default=1,
            help="After how many batches val step results have to be logged."
        )
        parser.add_argument(
            "--log_test_every_n_steps",
            type=int,
            default=1,
            help="After how many batches test step results have to be logged."
        )
        parser.add_argument("--reduce_lr_on_plateau_patience", type=int, help="patience for reduce_lr_on_plateau")
        return parent_parser
